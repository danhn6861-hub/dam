import streamlit as st
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from scipy.stats import entropy, zscore
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings("ignore")

# ------------------------------
# Utility functions (outliers / smoothing / WMA)
# ------------------------------

def handle_outliers(arr):
    """
    Safe outlier handling:
    - convert to numpy float
    - z-score clip with robust fallback
    - winsorize 5%-95%
    """
    arr = np.array(arr, dtype=float)
    if arr.size < 2:
        return arr.tolist()
    try:
        z = np.abs(zscore(arr, ddof=1))
        if np.isnan(z).all():
            z = np.zeros_like(arr)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        z = np.zeros_like(arr)
    median_val = float(np.nanmedian(arr))
    arr = np.array(arr, dtype=float)
    arr[z > 3] = median_val
    lower, upper = np.percentile(arr, [5, 95])
    arr[arr < lower] = lower
    arr[arr > upper] = upper
    return arr.tolist()


def ema_smoothing(arr, alpha=0.3):
    smoothed = []
    for i, val in enumerate(arr):
        if i == 0:
            smoothed.append(float(val))
        else:
            smoothed.append(float(alpha * val + (1 - alpha) * smoothed[-1]))
    return smoothed


def weighted_moving_average(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return 0.5
    weights = np.arange(1, arr.size + 1, dtype=float)
    return float(np.dot(arr, weights) / weights.sum())


# ------------------------------
# Feature engineering
# ------------------------------

def create_features(history, window=7):
    """
    Return X,y where each sample feature length = window + 7 extras:
    [window_slice (length=window), ent, momentum, streak, ratio_tai, wma, pattern_consistency, streak_strength]
    """
    encode = {"Tài": 1, "Xỉu": 0}
    history_num = [encode.get(h, 0) for h in history]
    history_smooth = ema_smoothing(history_num)
    X_list, y_list = [], []
    for i in range(window, len(history_num)):
        window_slice = history_smooth[i - window : i]
        window_slice = handle_outliers(window_slice)
        window_int = [int(round(v)) for v in window_slice]
        counts = np.bincount(np.array(window_int, dtype=int), minlength=2)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        try:
            ent = float(entropy(probs, base=2))
            if np.isnan(ent):
                ent = 1.0
        except Exception:
            ent = 1.0
        streak = 0
        for j in range(1, window + 1):
            if int(round(window_slice[-j])) == int(round(window_slice[-1])):
                streak += 1
            else:
                break
        momentum = float(np.mean(np.diff(window_slice[-3:]))) if len(window_slice) >= 3 else 0.0
        ratio_tai = float(counts[1] / counts.sum()) if counts.sum() > 0 else 0.5
        wma = float(weighted_moving_average(window_slice))
        pattern_consistency = float(np.std(window_slice))
        streak_strength = float(streak / window)
        ws = [float(v) for v in window_slice]
        if len(ws) < window:
            ws = [0.5] * (window - len(ws)) + ws
        features = ws + [ent, momentum, streak, ratio_tai, wma, pattern_consistency, streak_strength]
        X_list.append(features)
        y_list.append(history_num[i])
    if len(X_list) == 0:
        return np.empty((0, window + 7)), np.empty((0,))
    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


# ------------------------------
# Experts (return prob of T)
# ------------------------------

def expert_markov_prob(history):
    if len(history) < 1:
        return 0.5
    encode = {"Tài": 1, "Xỉu": 0}
    h = [encode.get(x, 0) for x in history]
    last = h[-1]
    next_counts = [1.0, 1.0]  # Laplace smoothing
    for i in range(len(h) - 1):
        if h[i] == last:
            nxt = h[i + 1]
            next_counts[int(nxt)] += 1.0
    prob_t = next_counts[1] / (next_counts[0] + next_counts[1])
    return float(prob_t)


def expert_freq_prob(history):
    if len(history) == 0:
        return 0.5
    count_t = sum(1 for x in history if x == "Tài")
    return float(count_t / len(history))


def expert_wma_prob(history, window=7):
    encode = {"Tài": 1, "Xỉu": 0}
    h = [encode.get(x, 0) for x in history[-window:]]
    if len(h) == 0:
        return 0.5
    return float(weighted_moving_average(h))


def expert_sgd_prob(sgd_model, history, window=7):
    if sgd_model is None:
        return expert_freq_prob(history)
    if len(history) < window:
        return expert_freq_prob(history)
    X_all, y_all = create_features(history, window)
    if X_all.shape[0] == 0:
        return expert_freq_prob(history)
    try:
        prob = float(sgd_model.predict_proba([X_all[-1]])[0][1])
    except Exception:
        prob = expert_freq_prob(history)
    return prob


def expert_constant_prob(history):
    return 0.5


# ------------------------------
# Noise metrics & helpers
# ------------------------------

def compute_noise_metrics(history, expert_probs, window=7):
    encode = {"Tài": 1, "Xỉu": 0}
    h = [encode.get(x, 0) for x in history]
    if len(h) == 0:
        return {"disagreement": 0.0, "label_entropy": 1.0, "streakiness": 0.0}
    recent_len = min(max(10, window * 2), len(h))
    recent = h[-recent_len:]
    p1 = (np.sum(recent) / len(recent)) if len(recent) > 0 else 0.5
    label_entropy = float(entropy([1 - p1, p1], base=2)) if p1 not in [0.0, 1.0] else 0.0
    disagreement = float(np.var(np.array(expert_probs, dtype=float))) if len(expert_probs) > 1 else 0.0
    # streakiness: last streak length / window
    streak = 1
    for i in range(len(recent) - 2, -1, -1):
        if recent[i] == recent[-1]:
            streak += 1
        else:
            break
    streakiness = float(min(streak, window) / max(window, 1))
    return {"disagreement": disagreement, "label_entropy": label_entropy, "streakiness": streakiness}


def choose_best_window(history, grid=(5, 7, 9, 11, 13)):
    # Simple prequential selection minimizing average log loss of WMA expert
    encode = {"Tài": 1, "Xỉu": 0}
    if len(history) < 10:
        return 7
    best_w, best_loss = 7, float("inf")
    for w in grid:
        if len(history) <= w:
            continue
        total_loss, count = 0.0, 0
        for i in range(1, len(history)):
            p = expert_wma_prob(history[:i], window=w)
            y = encode.get(history[i], 0)
            total_loss += log_loss(y, p)
            count += 1
        if count > 0:
            avg_loss = total_loss / count
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_w = w
    return best_w


# ------------------------------
# Hedge / Adaptive weights
# ------------------------------

def init_expert_state():
    names = ["markov", "freq", "wma", "sgd", "constant"]
    return {
        "names": names,
        "weights": np.ones(len(names), dtype=float),
        "eta": 0.5,
        "eta_min": 0.05,
        "eta_max": 2.0,
        "cum_variance": 1e-9,
        "rel_ema": np.ones(len(names), dtype=float),
        "rel_alpha": 0.1,
        "rel_gamma": 1.0,
        "adaptive": True,
    }


def hedge_update_adaptive(expert_state, losses):
    losses = np.array(losses, dtype=float)
    weights = expert_state["weights"].astype(float)
    K = len(weights)
    if weights.sum() <= 0 or np.isnan(weights.sum()):
        weights = np.ones_like(weights) / K
    else:
        weights = weights / weights.sum()

    # mix stats
    mix_loss = float(np.dot(weights, losses))
    variance = float(np.dot(weights, (losses - mix_loss) ** 2))
    expert_state["cum_variance"] = float(expert_state.get("cum_variance", 0.0) + max(variance, 0.0))

    # adaptive eta like AdaHedge
    if expert_state.get("adaptive", True):
        logK = max(math.log(K + 1e-12), 1e-9)
        eta = math.sqrt(2.0 * logK / (expert_state["cum_variance"] + 1e-12))
        eta = float(np.clip(eta, expert_state.get("eta_min", 0.05), expert_state.get("eta_max", 2.0)))
        expert_state["eta"] = eta
    eta = float(expert_state.get("eta", 0.5))

    # base exponential weighting
    new_w = expert_state["weights"] * np.exp(-eta * losses)

    # reliability fusion from loss EMA
    rel_ema = expert_state.get("rel_ema", np.ones_like(new_w))
    rel_alpha = float(expert_state.get("rel_alpha", 0.1))
    new_rel_ema = (1 - rel_alpha) * rel_ema + rel_alpha * losses
    gamma = float(expert_state.get("rel_gamma", 1.0))
    rel_factor = np.exp(-gamma * new_rel_ema)
    new_w = new_w * rel_factor

    # normalize
    total = new_w.sum()
    if total <= 0 or np.isnan(total):
        new_w = np.ones_like(new_w) / len(new_w)
    else:
        new_w = new_w / total

    expert_state["weights"] = new_w
    expert_state["rel_ema"] = new_rel_ema
    return expert_state


def log_loss(true_label, prob):
    eps = 1e-12
    p = float(np.clip(prob, eps, 1 - eps))
    return float(- (true_label * np.log(p) + (1 - true_label) * np.log(1 - p)))


# ------------------------------
# Combined predict: weighted experts + meta + noise-aware smoothing
# ------------------------------

def get_expert_probs_by_name(s, history, window):
    name_to_prob = {}
    for name in s["experts"]["names"]:
        if name == "markov":
            name_to_prob[name] = expert_markov_prob(history)
        elif name == "freq":
            name_to_prob[name] = expert_freq_prob(history)
        elif name == "wma":
            name_to_prob[name] = expert_wma_prob(history, window=window)
        elif name == "sgd":
            name_to_prob[name] = expert_sgd_prob(s.get("sgd_model", None), history, window=window)
        elif name == "constant":
            name_to_prob[name] = expert_constant_prob(history)
        else:
            name_to_prob[name] = 0.5
    probs_list = [float(name_to_prob[n]) for n in s["experts"]["names"]]
    return probs_list


def combined_predict(session_state, history, window=7, label_smoothing_alpha=0.1):
    s = session_state
    expert_probs = get_expert_probs_by_name(s, history, window)
    weights = s["experts"]["weights"].astype(float)
    if weights.sum() <= 0 or np.isnan(weights.sum()):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    hedge_prob = float(np.dot(weights, np.array(expert_probs, dtype=float)))

    noise = compute_noise_metrics(history, expert_probs, window=window)
    # dynamic label smoothing: stronger when disagreement high
    base_alpha = float(label_smoothing_alpha)
    alpha_dyn = base_alpha + 0.5 * max(0.0, noise["disagreement"] - 0.02)
    alpha_dyn = float(np.clip(alpha_dyn, 0.0, 0.3))
    hedge_prob_smoothed = float(alpha_dyn + (1 - 2 * alpha_dyn) * hedge_prob)

    # meta-learner stacking
    meta_prob = None
    if s.get("meta_model", None) is not None and s.get("meta_classes_initialized", False):
        try:
            meta_features = np.array([
                *expert_probs,
                noise["disagreement"],
                noise["label_entropy"],
                noise["streakiness"],
            ], dtype=float).reshape(1, -1)
            meta_prob = float(s["meta_model"].predict_proba(meta_features)[0][1])
        except Exception:
            meta_prob = None

    # blend hedge and meta using performance ema if available
    hedge_loss_ema = float(s.get("perf", {}).get("hedge_loss_ema", 0.69))
    meta_loss_ema = float(s.get("perf", {}).get("meta_loss_ema", 0.69))
    blend_w = 0.0
    if meta_prob is not None:
        # higher weight if meta beats hedge recently
        diff = hedge_loss_ema - meta_loss_ema
        blend_w = 1.0 / (1.0 + math.exp(-5.0 * diff))  # sigmoid in [0,1]
        blend_w = float(np.clip(blend_w, 0.1, 0.9))
    final_prob_raw = (1 - blend_w) * hedge_prob + blend_w * (meta_prob if meta_prob is not None else hedge_prob)

    # final smoothing again with dynamic alpha (conservative under noise)
    final_prob = float(alpha_dyn + (1 - 2 * alpha_dyn) * final_prob_raw)
    final_prob = float(np.clip(final_prob, 0.0, 1.0))

    return final_prob, {
        "expert_probs": expert_probs,
        "weights": weights,
        "hedge_prob": hedge_prob,
        "meta_prob": meta_prob,
        "noise": noise,
        "alpha_dyn": alpha_dyn,
    }


# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="🎯 Online Ensemble (Hedge) T/X Predictor", layout="wide")
st.title("🎯 Online Ensemble (Hedge) T/X Predictor — cho dữ liệu nhiễu cao (realtime)")

# session state init
if "history" not in st.session_state:
    st.session_state.history = []
if "window" not in st.session_state:
    st.session_state.window = 7
if "auto_window" not in st.session_state:
    st.session_state.auto_window = False
if "experts" not in st.session_state:
    st.session_state.experts = init_expert_state()
if "sgd_model" not in st.session_state:
    st.session_state.sgd_model = None
if "sgd_classes_initialized" not in st.session_state:
    st.session_state.sgd_classes_initialized = False
if "meta_model" not in st.session_state:
    st.session_state.meta_model = None
if "meta_classes_initialized" not in st.session_state:
    st.session_state.meta_classes_initialized = False
if "updated_until_len" not in st.session_state:
    st.session_state.updated_until_len = 0
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "rounds": [],
        "final_prob": [],
        "hedge_prob": [],
        "meta_prob": [],
        "real": [],
        "ensemble_loss": [],
        "noise_disagreement": [],
        "noise_label_entropy": [],
    }
if "perf" not in st.session_state:
    st.session_state.perf = {"hedge_loss_ema": 0.69, "meta_loss_ema": 0.69}

# --- Controls ---
st.subheader("1) Nhập kết quả (mới nhất cuối)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
with c2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
with c3:
    if st.button("Hoàn tác 1 ván"):
        if st.session_state.history:
            st.session_state.history.pop()
            st.session_state.updated_until_len = min(st.session_state.updated_until_len, len(st.session_state.history))
with c4:
    if st.button("Reset toàn bộ"):
        st.session_state.history = []
        st.session_state.experts = init_expert_state()
        st.session_state.sgd_model = None
        st.session_state.sgd_classes_initialized = False
        st.session_state.meta_model = None
        st.session_state.meta_classes_initialized = False
        st.session_state.updated_until_len = 0
        st.session_state.metrics = {
            "rounds": [],
            "final_prob": [],
            "hedge_prob": [],
            "meta_prob": [],
            "real": [],
            "ensemble_loss": [],
            "noise_disagreement": [],
            "noise_label_entropy": [],
        }
        st.session_state.perf = {"hedge_loss_ema": 0.69, "meta_loss_ema": 0.69}
        st.success("Đã reset mọi thứ.")

st.write("Số ván hiện có:", len(st.session_state.history))
st.write("Lịch sử (mới nhất cuối):", st.session_state.history)

# advanced controls
st.subheader("Thiết lập nâng cao")
colA, colB, colC, colD = st.columns(4)
with colA:
    st.session_state.auto_window = st.checkbox("Tự động chọn window", value=st.session_state.auto_window)
with colB:
    st.session_state.window = int(st.number_input("Window (nếu không auto)", min_value=3, max_value=25, value=int(st.session_state.window), step=2))
with colC:
    label_smoothing_alpha = float(st.slider("Label smoothing alpha", 0.0, 0.3, 0.10, 0.01))
with colD:
    abstain_threshold = float(st.slider("Ngưỡng tự tin để dự đoán (abstain)", 0.0, 0.25, 0.05, 0.01))

colE, colF = st.columns(2)
with colE:
    st.session_state.experts["adaptive"] = st.checkbox("Hedge: điều chỉnh eta tự động", value=st.session_state.experts.get("adaptive", True))
with colF:
    noise_skip_threshold = float(st.slider("Ngưỡng nhiễu (disagreement) để tạm dừng", 0.0, 0.5, 0.12, 0.01))

# dynamic window selection
if st.session_state.auto_window and len(st.session_state.history) >= 12:
    st.session_state.window = int(choose_best_window(st.session_state.history))

window = int(st.session_state.window)

# --- Prediction for next round ---
st.subheader("2) Dự đoán cho ván TIẾP THEO")
final_prob_next, aux = combined_predict(st.session_state, st.session_state.history, window=window, label_smoothing_alpha=label_smoothing_alpha)
expert_probs_next = aux["expert_probs"]
expert_weights = aux["weights"]
noise = aux["noise"]
pred_label_next = "Tài" if final_prob_next > 0.5 else "Xỉu"
confidence = abs(final_prob_next - 0.5)
should_skip = confidence < abstain_threshold or noise["disagreement"] > noise_skip_threshold
conf_status = "Đáng tin cậy ✅" if not should_skip and confidence >= 0.1 else ("Cân nhắc (trung bình) ⚠️" if not should_skip else "Khuyên bỏ qua (nhiễu cao) ⏸️")

st.markdown(f"**Xác suất Tài (final, noise-aware):** {final_prob_next:.2%} — Kết luận: **{pred_label_next}** | Trạng thái: {conf_status}")

if should_skip:
    st.info("Mức nhiễu cao hoặc độ tự tin thấp — gợi ý tạm dừng (abstain)")

# show experts
st.subheader("Phân tích experts (xác suất T)")
ep_cols = st.columns(len(st.session_state.experts["names"]))
for i, name in enumerate(st.session_state.experts["names"]):
    try:
        val = expert_probs_next[i]
        ep_cols[i].metric(name, f"{val:.2%}")
    except Exception:
        ep_cols[i].metric(name, "N/A")

# show weights and eta
st.subheader("Trọng số experts (Hedge) & eta")
wcols = st.columns(len(st.session_state.experts["names"]))
for i, name in enumerate(st.session_state.experts["names"]):
    try:
        wcols[i].metric(name, f"{expert_weights[i]:.3f}")
    except Exception:
        wcols[i].metric(name, "N/A")
st.caption(f"eta hiện tại: {st.session_state.experts.get('eta', 0.5):.3f} (tự động: {st.session_state.experts.get('adaptive', True)})")

# --- Online update logic (process new observations exactly once) ---
encode = {"Tài": 1, "Xỉu": 0}
if len(st.session_state.history) >= 1 and st.session_state.updated_until_len < len(st.session_state.history):
    # process rounds st.session_state.updated_until_len+1 .. len(history)
    for idx1 in range(st.session_state.updated_until_len + 1, len(st.session_state.history) + 1):
        i = idx1 - 1  # 0-based true index of observed label
        history_before = st.session_state.history[:i]
        if len(history_before) < 1:
            continue
        true_label = encode.get(st.session_state.history[i], 0)

        # expert probs BEFORE seeing the label
        probs_before = get_expert_probs_by_name(st.session_state, history_before, window)

        # compute mix prediction BEFORE update (with current weights)
        old_w = st.session_state.experts["weights"].astype(float)
        old_w = old_w / old_w.sum() if old_w.sum() > 0 else np.ones_like(old_w) / len(old_w)
        mix_prob_before = float(np.dot(old_w, np.array(probs_before, dtype=float)))
        # dynamic smoothing consistent with combined_predict
        noise_before = compute_noise_metrics(history_before, probs_before, window=window)
        alpha_dyn_before = float(np.clip(label_smoothing_alpha + 0.5 * max(0.0, noise_before["disagreement"] - 0.02), 0.0, 0.3))
        mix_prob_before_smoothed = float(alpha_dyn_before + (1 - 2 * alpha_dyn_before) * mix_prob_before)

        # compute per-expert losses and update hedge
        losses = [log_loss(true_label, p) for p in probs_before]
        st.session_state.experts = hedge_update_adaptive(st.session_state.experts, losses)

        # train/update SGD expert online using the new sample
        if len(st.session_state.history[:i + 1]) > window:
            X_batch, y_batch = create_features(st.session_state.history[:i + 1], window)
            if X_batch.shape[0] > 0:
                new_X = X_batch[-1].reshape(1, -1)
                new_y = np.array([y_batch[-1]])
                if st.session_state.sgd_model is None:
                    try:
                        sgd = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
                        sgd.partial_fit(new_X, new_y, classes=np.array([0, 1]))
                        st.session_state.sgd_model = sgd
                        st.session_state.sgd_classes_initialized = True
                    except Exception:
                        pass
                else:
                    try:
                        if not st.session_state.sgd_classes_initialized:
                            st.session_state.sgd_model.partial_fit(new_X, new_y, classes=np.array([0, 1]))
                            st.session_state.sgd_classes_initialized = True
                        else:
                            st.session_state.sgd_model.partial_fit(new_X, new_y)
                    except Exception:
                        pass

        # train/update meta-learner on expert probs and noise metrics
        meta_features = np.array([
            *probs_before,
            noise_before["disagreement"],
            noise_before["label_entropy"],
            noise_before["streakiness"],
        ], dtype=float).reshape(1, -1)
        if st.session_state.meta_model is None:
            try:
                meta = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=123)
                meta.partial_fit(meta_features, np.array([true_label]), classes=np.array([0, 1]))
                st.session_state.meta_model = meta
                st.session_state.meta_classes_initialized = True
            except Exception:
                pass
        else:
            try:
                if not st.session_state.meta_classes_initialized:
                    st.session_state.meta_model.partial_fit(meta_features, np.array([true_label]), classes=np.array([0, 1]))
                    st.session_state.meta_classes_initialized = True
                else:
                    st.session_state.meta_model.partial_fit(meta_features, np.array([true_label]))
            except Exception:
                pass

        # record metrics
        try:
            ens_loss = log_loss(true_label, mix_prob_before_smoothed)
            st.session_state.metrics["rounds"].append(i + 1)
            st.session_state.metrics["final_prob"].append(mix_prob_before_smoothed)
            st.session_state.metrics["hedge_prob"].append(mix_prob_before)
            # meta prob for logging
            try:
                meta_prob_log = float(st.session_state.meta_model.predict_proba(meta_features)[0][1]) if st.session_state.meta_model is not None and st.session_state.meta_classes_initialized else None
            except Exception:
                meta_prob_log = None
            st.session_state.metrics["meta_prob"].append(meta_prob_log if meta_prob_log is not None else np.nan)
            st.session_state.metrics["real"].append(true_label)
            st.session_state.metrics["ensemble_loss"].append(ens_loss)
            st.session_state.metrics["noise_disagreement"].append(noise_before["disagreement"])
            st.session_state.metrics["noise_label_entropy"].append(noise_before["label_entropy"])

            # update perf ema for hedge/meta
            beta = 0.1
            st.session_state.perf["hedge_loss_ema"] = (1 - beta) * st.session_state.perf.get("hedge_loss_ema", 0.69) + beta * ens_loss
            if meta_prob_log is not None and not np.isnan(meta_prob_log):
                st.session_state.perf["meta_loss_ema"] = (1 - beta) * st.session_state.perf.get("meta_loss_ema", 0.69) + beta * log_loss(true_label, meta_prob_log)
        except Exception:
            pass

    st.session_state.updated_until_len = len(st.session_state.history)

# --- Diagnostics & charts ---
if len(st.session_state.metrics["rounds"]) > 0:
    st.subheader("Đường cong học / Log loss của ensemble theo ván")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state.metrics["rounds"], st.session_state.metrics["ensemble_loss"], marker='o')
    ax.set_xlabel("Ván")
    ax.set_ylabel("Log loss (ensemble)")
    ax.set_title("Ensemble log loss theo ván (online update)")
    st.pyplot(fig)

    st.subheader("Nhiễu: Disagreement & Label entropy")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(st.session_state.metrics["rounds"], st.session_state.metrics["noise_disagreement"], label="disagreement")
    ax2.plot(st.session_state.metrics["rounds"], st.session_state.metrics["noise_label_entropy"], label="label_entropy")
    ax2.legend()
    ax2.set_xlabel("Ván")
    st.pyplot(fig2)

# --- Quick stats ---
if len(st.session_state.history) > 0:
    st.subheader("Thống kê nhanh")
    total = len(st.session_state.history)
    ct = sum(1 for x in st.session_state.history if x == "Tài")
    st.write(f"Tổng ván: {total} | Tài: {ct} ({ct/total:.2%}) | Xỉu: {total-ct} ({(total-ct)/total:.2%})")

# --- Optional: train heavy batch models if user requests ---
st.subheader("Tùy chọn: Huấn luyện batch models (RF, XGB, SVM) — chỉ khi đủ dữ liệu")
if st.button("Huấn luyện batch models (RF, XGB, SVM)"):
    if len(st.session_state.history) > window + 5:
        Xb, yb = create_features(st.session_state.history, window)
        if Xb.shape[0] > 5:
            try:
                model_lr = LogisticRegression(C=0.5, solver='liblinear', random_state=42).fit(Xb, yb)
                model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(Xb, yb)
                model_xgb = None
                if HAS_XGB:
                    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(Xb, yb)
                model_svm = SVC(probability=True, kernel='rbf', random_state=42).fit(Xb, yb)
                st.session_state.batch_models = [model_lr, model_rf, model_xgb, model_svm]
                st.success("Huấn luyện batch xong.")
            except Exception as e:
                st.error(f"Lỗi khi huấn luyện batch: {e}")
        else:
            st.info("Không đủ mẫu để huấn luyện batch.")
    else:
        st.info("Cần thêm dữ liệu để huấn luyện batch.")

st.markdown(
    """
**Ghi chú / hướng tối ưu thêm**
- Hedge (exponential weights) sử dụng eta tự điều chỉnh + độ tin cậy theo log-loss EMA, bền với nhiễu.
- Meta-learner học trực tiếp từ vector xác suất experts + đặc trưng nhiễu (disagreement, label entropy, streakiness).
- Cơ chế abstention giúp tránh quyết định khi tín hiệu quá yếu hoặc nhiễu quá cao.
- Có thể tune: eta_min/max, rel_gamma, label_smoothing, window, ngưỡng abstain/noise.
- "Tự động chọn window" dùng đánh giá prequential của WMA để chọn W tối ưu đơn giản.
"""
)
