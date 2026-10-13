/* ══════════════════════════════════════════════════════════
   NEURA-LINK — 前端邏輯
   影像串流 + 狀態輪詢 + 馬達控制 + 線上人臉註冊
   ══════════════════════════════════════════════════════════ */

'use strict';

/* ─────────── DOM ─────────── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const el = {
    fps:            $('#fps'),
    uplink:         $('#uplink'),
    registered:     $('#registered-count'),
    faceCount:      $('#face-count'),
    knownCount:     $('#known-count'),
    threshold:      $('#threshold'),
    faceTag:        $('#face-tag'),
    motorAction:    $('#motor-action'),
    motorWheels:    $('#motor-wheels'),
    speed:          $('#speed'),
    speedVal:       $('#speed-val'),
    servoX:         $('#servo-x'),
    servoY:         $('#servo-y'),
    servoXVal:      $('#servo-x-val'),
    servoYVal:      $('#servo-y-val'),
    regLabel:       $('#reg-label'),
    regCount:       $('#reg-count'),
    regProgress:    $('#reg-progress'),
    regName:        $('#reg-name'),
    regCountText:   $('#reg-count-text'),
    regFill:        $('#reg-fill'),
    regBadge:       $('#reg-badge'),
    peopleUl:       $('#people-ul'),
    peopleTotal:    $('#people-total'),
    logBody:        $('#log-body'),
    toast:          $('#toast'),
};

const KEY_ACTIONS = {
    'ArrowUp':    'forward',
    'w':          'forward',
    'ArrowDown':  'backward',
    's':          'backward',
    'ArrowLeft':  'left',
    'a':          'left',
    'ArrowRight': 'right',
    'd':          'right',
    ' ':          'stop',
};

/* ─────────── 工具 ─────────── */

function log(msg, level = 'info') {
    if (!el.logBody) return;
    const line = document.createElement('div');
    line.className = `log-line ${level}`;

    const now = new Date();
    const t = [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map(n => String(n).padStart(2, '0')).join(':');

    line.innerHTML = `<span class="log-time">${t}</span><span class="log-msg"></span>`;
    line.querySelector('.log-msg').textContent = msg;

    el.logBody.appendChild(line);
    el.logBody.scrollTop = el.logBody.scrollHeight;

    // 只保留最近 200 行
    while (el.logBody.children.length > 200) {
        el.logBody.removeChild(el.logBody.firstChild);
    }
}

let toastTimer = null;
function toast(msg, type = '') {
    if (!el.toast) return;
    el.toast.textContent = msg;
    el.toast.className = `toast show ${type}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        el.toast.className = `toast ${type}`;
    }, 3000);
}

async function api(url, options = {}) {
    try {
        const res = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(data.error || `HTTP ${res.status}`);
        }
        return data;
    } catch (err) {
        // silent=true 時不噴錯誤（例如「尚未訓練模型」屬正常狀態）
        if (!options.silent) {
            log(`API 錯誤：${err.message}`, 'err');
            toast(err.message, 'error');
        }
        throw err;
    }
}

/* ─────────── 狀態輪詢 ─────────── */

let lastFaceKey = '';

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const s = await res.json();

        // FPS
        el.fps.textContent = s.fps != null ? s.fps.toFixed(1) : '--';

        // UPLINK
        if (el.uplink) {
            const running = s.running;
            el.uplink.innerHTML = running
                ? '<span class="dot"></span>ONLINE'
                : '<span class="dot" style="background:#ff3b3b;box-shadow:0 0 8px #ff3b3b"></span>OFFLINE';
        }

        // 統計
        el.registered.textContent = (s.labels || []).length;
        el.faceCount.textContent  = s.face_count;
        el.knownCount.textContent = s.known_count;
        el.threshold.textContent = `${s.threshold_strong} / ${s.threshold_weak}`;

        // 臉部標籤
        const faces = s.faces || [];
        if (faces.length === 0) {
            el.faceTag.textContent = s.model_loaded ? 'NO FACE DETECTED' : 'NO MODEL — 請先註冊並訓練';
            el.faceTag.className = 'face-tag';
        } else {
            const names = faces.map(f => {
                const d = f.distance != null ? `(${f.distance})` : '';
                return `${f.name}${d}`;
            }).join('  ');
            el.faceTag.textContent = names;
            el.faceTag.className = 'face-tag' +
                (s.known_count > 0 ? ' known' : (s.face_count > 0 ? ' partial' : ''));

            // 只在辨識結果變化時記錄，避免刷版
            const key = names;
            if (key !== lastFaceKey) {
                lastFaceKey = key;
                if (s.known_count > 0) {
                    log(`辨識到：${faces.filter(f => f.is_known).map(f => f.name).join(', ')}`, 'ok');
                }
            }
        }

        // 馬達狀態
        if (s.motor) {
            const m = s.motor;
            el.motorAction.textContent = m.label;
            el.motorWheels.textContent = `L ${m.left_speed} · R ${m.right_speed}`;

            $$('[data-action]').forEach(b => {
                b.classList.toggle('active', b.dataset.action === m.action);
            });
        }

        // 註冊進度
        const btnRegister  = $('#btn-register');
        const btnCancelReg = $('#btn-cancel-reg');

        if (s.register_task) {
            const t = s.register_task;
            el.regProgress.classList.remove('hidden');
            el.regName.textContent = t.label;
            el.regCountText.textContent = `${t.count} / ${t.target}`;
            const pct = Math.min(100, (t.count / Math.max(1, t.target)) * 100);
            el.regFill.style.width = `${pct}%`;
            el.regBadge.textContent = 'CAPTURING';
            el.regBadge.className = 'panel-badge active';

            // 註冊進行中才顯示取消鈕，並鎖住開始鈕避免重複送出
            btnCancelReg.classList.remove('hidden');
            btnRegister.disabled = true;
            btnRegister.textContent = '擷取中…';
        } else {
            if (!el.regProgress.classList.contains('hidden')) {
                // 剛好擷取完成
                el.regProgress.classList.add('hidden');
                el.regBadge.textContent = 'IDLE';
                el.regBadge.className = 'panel-badge';
                log('擷取完成，請按「重新訓練模型」', 'ok');
                toast('擷取完成，請記得重新訓練模型', 'ok');
                refreshPeople();
            }
            btnCancelReg.classList.add('hidden');
            btnRegister.disabled = false;
            btnRegister.textContent = '開始註冊';
        }

    } catch (err) {
        // 伺服器可能正在重啟，靜默重試即可
        if (el.uplink) {
            el.uplink.innerHTML =
                '<span class="dot" style="background:#ff3b3b;box-shadow:0 0 8px #ff3b3b"></span>OFFLINE';
        }
    }
}

/* ─────────── 馬達控制 ─────────── */

async function sendAction(action) {
    try {
        const data = await api(`/api/motor/${action}`, {
            method: 'POST',
            body: JSON.stringify({ speed: parseInt(el.speed.value, 10) }),
        });
        const m = data.motor;
        el.motorAction.textContent = m.label;
        el.motorWheels.textContent = `L ${m.left_speed} · R ${m.right_speed}`;
        $$('[data-action]').forEach(b => {
            b.classList.toggle('active', b.dataset.action === action);
        });
    } catch (err) {
        /* 錯誤已由 api() 顯示 */
    }
}

function bindDpad() {
    // 注意：用 [data-action] 而不是 .dbtn。
    // STOP 按鈕的 class 是 .btn 不是 .dbtn，只綁 .dbtn 會漏掉它。
    $$('[data-action]').forEach(btn => {
        const action = btn.dataset.action;

        // 滑鼠：按下即送出
        btn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            sendAction(action);
        });

        // 觸控
        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            sendAction(action);
        }, { passive: false });

        // 鍵盤可及性
        btn.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                sendAction(action);
            }
        });

        // 鍵盤操作時沒有 mousedown，靠 click 補一次
        btn.addEventListener('click', (e) => {
            if (e.detail === 0) {        // detail===0 代表不是真實滑鼠點擊
                sendAction(action);
            }
        });
    });

    // 緊急停止
    $('#btn-emergency').addEventListener('click', async () => {
        try {
            const d = await api('/api/motor/emergency', { method: 'POST' });
            log('緊急停止！', 'err');
            toast('緊急停止已執行', 'error');
            $$('[data-action]').forEach(b => b.classList.remove('active'));
        } catch (err) { /* 已處理 */ }
    });
}

/* ─────────── 滑桿 ─────────── */

function bindSliders() {
    // 速度
    el.speed.addEventListener('input', () => {
        el.speedVal.textContent = `${el.speed.value}%`;
    });
    el.speed.addEventListener('change', async () => {
        try {
            await api('/api/motor/speed', {
                method: 'POST',
                body: JSON.stringify({ speed: parseInt(el.speed.value, 10) }),
            });
            log(`速度調整為 ${el.speed.value}%`);
        } catch (err) { /* 已處理 */ }
    });

    // 雲台
    function sendServo() {
        return api('/api/motor/servo', {
            method: 'POST',
            body: JSON.stringify({
                x: parseInt(el.servoX.value, 10),
                y: parseInt(el.servoY.value, 10),
            }),
        });
    }

    el.servoX.addEventListener('input', () => {
        el.servoXVal.textContent = `${el.servoX.value}°`;
    });
    el.servoY.addEventListener('input', () => {
        el.servoYVal.textContent = `${el.servoY.value}°`;
    });
    el.servoX.addEventListener('change', sendServo);
    el.servoY.addEventListener('change', sendServo);
}

/* ─────────── 人臉註冊 ─────────── */

function bindRegister() {
    $('#btn-register').addEventListener('click', async () => {
        const label = el.regLabel.value.trim();
        if (!label) {
            toast('請先輸入名稱', 'error');
            return;
        }

        const btn = $('#btn-register');
        btn.disabled = true;
        btn.textContent = '啟動中…';

        try {
            await api('/api/register/start', {
                method: 'POST',
                body: JSON.stringify({
                    label,
                    count: parseInt(el.regCount.value, 10) || 30,
                }),
            });
            log(`開始註冊「${label}」，請看著鏡頭`, 'info');
            toast('註冊開始，請看著鏡頭', 'ok');
            // 立即顯示進度條與取消鈕，不必等輪詢
            el.regProgress.classList.remove('hidden');
            $('#btn-cancel-reg').classList.remove('hidden');
        } catch (err) {
            btn.disabled = false;
            btn.textContent = '開始註冊';
        }
    });

    $('#btn-cancel-reg').addEventListener('click', async () => {
        try {
            await api('/api/register/cancel', { method: 'POST' });
            log('已取消註冊', 'warn');
            toast('已取消註冊');
            el.regProgress.classList.add('hidden');
            $('#btn-cancel-reg').classList.add('hidden');
            $('#btn-register').disabled = false;
            $('#btn-register').textContent = '開始註冊';
        } catch (err) { /* 已處理 */ }
    });

    $('#btn-train').addEventListener('click', async () => {
        try {
            const d = await api('/api/train', { method: 'POST' });
            log('開始重新訓練模型…', 'info');
            toast(d.message || '訓練中…', 'ok');
            // 訓練在背景執行，幾秒後刷新清單
            setTimeout(refreshPeople, 2500);
        } catch (err) { /* 已處理 */ }
    });

    $('#btn-snapshot').addEventListener('click', () => {
        // 直接開新視窗下載目前畫面
        const a = document.createElement('a');
        a.href = '/video_snapshot';
        a.download = `neuralink_${Date.now()}.jpg`;
        a.click();
        log('已擷取畫面快照', 'ok');
        toast('快照已下載', 'ok');
    });
}

/* ─────────── 已註冊清單 ─────────── */

async function refreshPeople() {
    try {
        const res = await fetch('/api/people');
        const d = await res.json();
        const people = d.people || [];

        el.peopleTotal.textContent = people.length;

        if (people.length === 0) {
            el.peopleUl.innerHTML = '<li class="empty">尚無註冊資料</li>';
            return;
        }

        el.peopleUl.innerHTML = '';
        people.forEach(p => {
            const li = document.createElement('li');

            const name = document.createElement('span');
            name.className = 'people-name';
            name.textContent = p.label;

            const right = document.createElement('span');
            const cnt = document.createElement('span');
            cnt.className = 'people-count';
            cnt.textContent = `${p.count} 張`;

            const del = document.createElement('button');
            del.className = 'people-del';
            del.textContent = '×';
            del.title = '刪除此身分';
            del.addEventListener('click', async () => {
                if (!confirm(`確定要刪除「${p.label}」的所有照片嗎？`)) return;
                try {
                    const r = await api(`/api/delete/${encodeURIComponent(p.label)}`,
                                       { method: 'POST' });
                    log(`已刪除 ${p.label}`, 'warn');
                    toast(r.message);
                    refreshPeople();
                } catch (err) { /* 已處理 */ }
            });

            right.appendChild(cnt);
            right.appendChild(del);
            li.appendChild(name);
            li.appendChild(right);
            el.peopleUl.appendChild(li);
        });
    } catch (err) {
        /* 已處理 */
    }
}

/* ─────────── 鍵盤遙控 ─────────── */

function bindKeyboard() {
    document.addEventListener('keydown', (e) => {
        // 在輸入框內打字時不要觸發遙控
        const tag = (e.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;

        const action = KEY_ACTIONS[e.key] || KEY_ACTIONS[e.key.toLowerCase()];
        if (action) {
            e.preventDefault();
            sendAction(action);
            log(`鍵盤指令：${action}`);
        }

        // r = 重新載入模型
        if (e.key.toLowerCase() === 'r') {
            loadModel();
        }
    });
}

async function loadModel() {
    try {
        const d = await api('/api/load_model', { method: 'POST', silent: true });
        log(`模型已載入：${d.labels.join(', ')}`, 'ok');
        toast(`已載入 ${d.labels.length} 人`, 'ok');
    } catch (err) {
        // 還沒訓練過模型是正常狀態，不當成錯誤
        log('尚無訓練模型，請先註冊人臉', 'warn');
    }
}

/* ─────────── 影像串流 ─────────── */

function bindVideo() {
    const img = $('#video');

    // 串流中斷時自動重連
    img.addEventListener('error', () => {
        setTimeout(() => {
            img.src = `/video_feed?t=${Date.now()}`;
        }, 1200);
    });

    // 頁面隱藏時暫停輪詢以省資源
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearInterval(pollTimer);
            pollTimer = null;
        } else if (!pollTimer) {
            pollTimer = setInterval(pollStatus, 400);
            pollStatus();
        }
    });
}

/* ─────────── 啟動 ─────────── */

let pollTimer = null;

function init() {
    log('系統初始化中…', 'info');

    bindDpad();
    bindSliders();
    bindRegister();
    bindKeyboard();
    bindVideo();

    $('#btn-clear-log').addEventListener('click', () => {
        el.logBody.innerHTML = '';
        log('日誌已清除');
    });

    // 先嘗試載入既有模型
    loadModel();
    refreshPeople();

    // 狀態輪詢
    pollStatus();
    pollTimer = setInterval(pollStatus, 400);

    log('系統就緒', 'ok');
    console.log('[NEURA-LINK] 初始化完成');
}

document.addEventListener('DOMContentLoaded', init);
