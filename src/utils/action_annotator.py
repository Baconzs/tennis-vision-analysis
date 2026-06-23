"""action_annotator.py — 网球动作时序标注工具（Flask Web 应用）

功能：从 rallies_new 按视频来源轮换提取片段到标注工作区，支持进度持久化
用法：python action_annotator.py，然后访问 http://localhost:5000
"""
import os
import json
import shutil
from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

SOURCE_FOLDER = os.path.join(_PROJECT_DIR, "data", "rallies_new")
WORK_FOLDER   = os.path.join(_PROJECT_DIR, "data", "rallies_annotating")
PROGRESS_FILE = os.path.join(WORK_FOLDER, "_progress.json")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    os.makedirs(WORK_FOLDER, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_source_folders():
    if not os.path.exists(SOURCE_FOLDER):
        return []
    return sorted([
        d for d in os.listdir(SOURCE_FOLDER)
        if os.path.isdir(os.path.join(SOURCE_FOLDER, d))
    ])


def get_source_rallies(source_name):
    src_dir = os.path.join(SOURCE_FOLDER, source_name)
    return sorted([
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
        and os.path.exists(os.path.join(src_dir, d, "raw_clip.mp4"))
    ])


def get_work_rallies(source_name):
    work_src_dir = os.path.join(WORK_FOLDER, source_name)
    if not os.path.exists(work_src_dir):
        return []
    return sorted([
        d for d in os.listdir(work_src_dir)
        if os.path.isdir(os.path.join(work_src_dir, d))
        and os.path.exists(os.path.join(work_src_dir, d, "raw_clip.mp4"))
    ])


def get_deleted_set(source_name):
    progress = load_progress()
    return set(progress.get("deleted", {}).get(source_name, []))


def mark_deleted(source_name, rally_name):
    progress = load_progress()
    deleted = progress.setdefault("deleted", {})
    deleted.setdefault(source_name, [])
    if rally_name not in deleted[source_name]:
        deleted[source_name].append(rally_name)
    save_progress(progress)


def _transcode_to_h264(src_mp4, dst_mp4):
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', src_mp4,
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac', '-movflags', '+faststart',
        dst_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_next_rally(source_name):
    all_rallies = get_source_rallies(source_name)
    already = set(get_work_rallies(source_name)) | get_deleted_set(source_name)
    for rally_name in all_rallies:
        if rally_name not in already:
            src_path = os.path.join(SOURCE_FOLDER, source_name, rally_name)
            dst_path = os.path.join(WORK_FOLDER, source_name, rally_name)
            os.makedirs(os.path.join(WORK_FOLDER, source_name), exist_ok=True)
            shutil.copytree(src_path, dst_path)
            mp4 = os.path.join(dst_path, 'raw_clip.mp4')
            tmp = mp4 + '.tmp.mp4'
            try:
                _transcode_to_h264(mp4, tmp)
                os.replace(tmp, mp4)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return rally_name
    return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>网球动作时序标注</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f5f7fa; display: flex; gap: 20px; height: 90vh; }
        .sidebar { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow-y: auto; min-width: 220px; }
        .main-panel { flex: 3; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }
        .video-container, .data-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        video { width: 100%; max-height: 50vh; background: #000; outline: none; border-radius: 4px; }
        .video-wrapper { position: relative; display: inline-block; width: 100%; }
        #actionOverlay { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.55); color: #fff; font-size: 18px; font-weight: bold; padding: 6px 14px; border-radius: 6px; pointer-events: none; display: none; }
        .source-group { margin-bottom: 12px; }
        .source-header { font-weight: bold; font-size: 12px; color: #555; padding: 5px 4px; border-bottom: 2px solid #e0e0e0; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; gap: 4px; }
        .source-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .btn-extract { font-size: 11px; padding: 2px 8px; background: #67c23a; color: white; border: none; border-radius: 3px; cursor: pointer; white-space: nowrap; }
        .btn-extract:hover { background: #4fa82a; }
        .playlist-item { padding: 7px 10px; border-bottom: 1px solid #eee; cursor: pointer; border-radius: 4px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; }
        .playlist-item:hover { background-color: #f0f7ff; }
        .playlist-item.active { background-color: #e1f0ff; border-left: 4px solid #409eff; font-weight: bold; }
        .badge { font-size: 11px; padding: 2px 6px; border-radius: 10px; color: white; }
        .badge-has-json { background: #67c23a; }
        .badge-no-json { background: #909399; }
        .controls { margin-top: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        button { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; background: #409eff; color: white; }
        button:hover { background: #66b1ff; }
        .btn-speed { background: #e6a23c; }
        .btn-speed.active { background: #b57a24; font-weight: bold; }
        .btn-danger { background: #f56c6c; font-size: 12px; }
        .btn-delete-clip { background: #c0392b; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; text-align: center; }
        th, td { border: 1px solid #ebeef5; padding: 8px; }
        th { background-color: #f8f9fa; }
        tr.selected-row td { background-color: #fff3cd; outline: 2px solid #e6a23c; }
    </style>
</head>
<body>
<div class="sidebar">
    <h3 style="margin-top:0;">片段列表</h3>
    <button onclick="loadPlaylist()" style="margin-bottom:12px; width:100%; background:#67c23a;">刷新目录</button>
    <div id="playlist">加载中...</div>
</div>

<div class="main-panel">
    <div class="video-container">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h2 id="currentVideoTitle" style="margin:0; color:#303133; font-size:16px;">请选择片段</h2>
            <button id="btnDeleteClip" class="btn-delete-clip" onclick="deleteClip()" style="display:none;">删除此片段</button>
        </div>
        <div class="video-wrapper">
            <video id="videoPlayer" controls></video>
            <div id="actionOverlay"></div>
        </div>
        <div class="controls">
            <button onclick="setPoint('start')">手动设起点 [Q]</button>
            <button onclick="setPoint('end')">设为终点并暂停 [E]</button>
            <span style="margin-left:15px; font-size:14px;">
                当前区间: <span id="lblStart" style="color:#409eff;font-weight:bold;">0.000</span> →
                <span id="lblEnd" style="color:#f56c6c;font-weight:bold;">未定</span>
            </span>
            <span style="margin-left:auto;">
                播放速度:
                <button id="btnSpeed05" class="btn-speed active" onclick="setSpeed(0.5)">0.5x</button>
                <button id="btnSpeed10" class="btn-speed" onclick="setSpeed(1.0)">1.0x</button>
            </span>
        </div>
        <div class="controls" style="margin-top:15px;">
            <button onclick="addAction('正手',1)" style="background:#67c23a;">正手 [1]</button>
            <button onclick="addAction('反手',2)" style="background:#e6a23c;">反手 [2]</button>
            <button onclick="addAction('发球',3)" style="background:#f56c6c;">发球 [3]</button>
            <button onclick="addAction('移动',4)" style="background:#909399;">移动 [4]</button>
            <button onclick="addAction('待机',0)" style="background:#303133;">待机 [5]</button>
        </div>
    </div>
    <div class="data-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0;">标注数据</h3>
            <span id="saveStatus" style="color:#67c23a; font-weight:bold; display:none;">已自动保存</span>
        </div>
        <table id="annotationTable">
            <thead><tr><th>起 (秒)</th><th>止 (秒)</th><th>动作</th><th>操作</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
</div>
<script>
    let currentSource = null;
    let currentFolder = null;
    let annotations = [];
    let tempStart = 0.000;
    let tempEnd = null;
    let selectedRowIdx = null;
    const video = document.getElementById('videoPlayer');

    async function loadPlaylist() {
        const res = await fetch('/api/list');
        const data = await res.json();
        const playlistDiv = document.getElementById('playlist');
        playlistDiv.innerHTML = '';

        data.forEach(src => {
            const group = document.createElement('div');
            group.className = 'source-group';

            const header = document.createElement('div');
            header.className = 'source-header';
            const btn = document.createElement('button');
            btn.className = 'btn-extract';
            btn.textContent = '提取';
            btn.dataset.source = src.source_name;
            btn.addEventListener('click', () => extractOne(src.source_name));
            header.innerHTML = '<span class="source-name" title="' + src.source_name + '">' + src.source_name + '</span>';
            header.appendChild(btn);
            group.appendChild(header);

            src.rallies.forEach(r => {
                const item = document.createElement('div');
                item.className = 'playlist-item';
                const badge = r.has_json
                    ? '<span class="badge badge-has-json">已标</span>'
                    : '<span class="badge badge-no-json">未标</span>';
                item.innerHTML = '<span>' + r.rally_name + '</span>' + badge;
                item.onclick = () => loadVideo(src.source_name, r.rally_name, item);
                group.appendChild(item);
            });

            playlistDiv.appendChild(group);
        });
    }

    async function extractOne(sourceName) {
        const res = await fetch('/api/extract/' + encodeURIComponent(sourceName), {method:'POST'});
        const data = await res.json();
        if (data.extracted) {
            await loadPlaylist();
        } else {
            alert(sourceName + ' 已全部提取完毕');
        }
    }

    async function loadVideo(sourceName, folderName, element) {
        document.querySelectorAll('.playlist-item').forEach(el => el.classList.remove('active'));
        if (element) element.classList.add('active');

        currentSource = sourceName;
        currentFolder = folderName;
        document.getElementById('currentVideoTitle').innerText = sourceName + ' / ' + folderName;
        document.getElementById('btnDeleteClip').style.display = 'inline-block';

        video.src = '/api/video/' + encodeURIComponent(sourceName) + '/' + encodeURIComponent(folderName);
        setSpeed(0.5);

        const res = await fetch('/api/json/' + encodeURIComponent(sourceName) + '/' + encodeURIComponent(folderName));
        const data = await res.json();
        annotations = data.annotations || [];
        renderTable();

        tempStart = annotations.length > 0 ? annotations[annotations.length - 1].end_time : 0.000;
        tempEnd = null;
        updateUI();
        selectedRowIdx = null;
        video.play().catch(e => console.log('等待交互'));
    }

    async function deleteClip() {
        if (!currentFolder || !currentSource) return;
        if (!confirm('确认删除片段 ' + currentFolder + '？删除后将自动从同一视频提取下一个片段。')) return;

        const res = await fetch('/api/delete/' + encodeURIComponent(currentSource) + '/' + encodeURIComponent(currentFolder), {method:'DELETE'});
        const data = await res.json();

        video.src = '';
        document.getElementById('currentVideoTitle').innerText = '请选择片段';
        document.getElementById('btnDeleteClip').style.display = 'none';
        annotations = [];
        renderTable();
        currentFolder = null;
        currentSource = null;

        await loadPlaylist();

        if (data.next_extracted) {
            alert('已删除，并从同一视频提取了新片段：' + data.next_extracted);
        } else {
            alert('已删除。该视频来源已无更多片段可提取。');
        }
    }

    function setSpeed(speed) {
        video.playbackRate = speed;
        document.getElementById('btnSpeed05').classList.toggle('active', speed === 0.5);
        document.getElementById('btnSpeed10').classList.toggle('active', speed === 1.0);
    }

    function setPoint(type) {
        if (!currentFolder) return;
        const time = parseFloat(video.currentTime.toFixed(3));
        if (type === 'start') {
            tempStart = time;
        } else {
            video.pause();
            tempEnd = time;
        }
        updateUI();
    }

    function addAction(name, id) {
        if (tempStart === null || tempEnd === null || tempStart >= tempEnd) return;
        annotations.push({ start_time: tempStart, end_time: tempEnd, action_name: name, action_id: id });
        annotations.sort((a, b) => a.start_time - b.start_time);
        tempStart = tempEnd;
        tempEnd = null;
        updateUI();
        renderTable();
        saveToServer();
    }

    window.deleteAction = function(idx) {
        annotations.splice(idx, 1);
        if (idx === annotations.length && annotations.length > 0) {
            tempStart = annotations[annotations.length - 1].end_time;
            updateUI();
        } else if (annotations.length === 0) {
            tempStart = 0.000;
            updateUI();
        }
        renderTable();
        saveToServer();
    };

    function updateUI() {
        document.getElementById('lblStart').innerText = tempStart !== null ? tempStart.toFixed(3) : '未定';
        document.getElementById('lblEnd').innerText = tempEnd !== null ? tempEnd.toFixed(3) : '未定';
    }

    function renderTable() {
        const tbody = document.querySelector('#annotationTable tbody');
        tbody.innerHTML = '';
        annotations.forEach((item, idx) => {
            const tr = document.createElement('tr');
            if (idx === selectedRowIdx) tr.classList.add('selected-row');
            tr.style.cursor = 'pointer';
            tr.innerHTML = '<td>' + item.start_time.toFixed(3) + '</td><td>' + item.end_time.toFixed(3) + '</td><td>' + item.action_name + '</td>' +
                '<td><button class="btn-danger" onclick="deleteAction(' + idx + ')">删除</button></td>';
            tr.onclick = (e) => {
                if (e.target.tagName === 'BUTTON') return;
                selectedRowIdx = (selectedRowIdx === idx) ? null : idx;
                renderTable();
            };
            tbody.appendChild(tr);
        });
    }

    async function saveToServer() {
        if (!currentFolder || !currentSource) return;
        const res = await fetch('/api/save/' + encodeURIComponent(currentSource) + '/' + encodeURIComponent(currentFolder), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(annotations)
        });
        if (res.ok) {
            const status = document.getElementById('saveStatus');
            status.style.display = 'block';
            setTimeout(() => status.style.display = 'none', 1500);
            loadPlaylist();
        }
    }

    const ACTION_MAP = { '1': ['正手',1], '2': ['反手',2], '3': ['发球',3], '4': ['移动',4], '5': ['待机',0] };
    document.addEventListener('keydown', function(e) {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const k = e.key.toLowerCase();
        if (k === 'q') { setPoint('start'); e.preventDefault(); }
        if (k === 'e') { setPoint('end'); e.preventDefault(); }
        if (ACTION_MAP[k]) {
            const [name, id] = ACTION_MAP[k];
            if (selectedRowIdx !== null) {
                annotations[selectedRowIdx].action_name = name;
                annotations[selectedRowIdx].action_id = id;
                selectedRowIdx = null;
                renderTable();
                saveToServer();
            } else {
                addAction(name, id);
            }
        }
        if (k === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    });

    const ACTION_COLORS = { '正手': '#67c23a', '反手': '#e6a23c', '发球': '#f56c6c', '移动': '#909399', '待机': '#303133' };
    video.addEventListener('timeupdate', function() {
        const t = video.currentTime;
        const overlay = document.getElementById('actionOverlay');
        const hit = annotations.find(a => t >= a.start_time && t < a.end_time);
        if (hit) {
            overlay.innerText = hit.action_name;
            overlay.style.background = (ACTION_COLORS[hit.action_name] || '#555') + 'cc';
            overlay.style.display = 'block';
        } else {
            overlay.style.display = 'none';
        }
    });

    loadPlaylist();
</script>
</body>
</html>
"""



# ─── Flask 路由 ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/list')
def get_video_list():
    result = []
    for source_name in get_source_folders():
        rallies = []
        for rally_name in get_work_rallies(source_name):
            work_dir = os.path.join(WORK_FOLDER, source_name, rally_name)
            json_path = os.path.join(work_dir, "annotations.json")
            rallies.append({
                "rally_name": rally_name,
                "has_json": os.path.exists(json_path) and os.path.getsize(json_path) > 5
            })
        result.append({"source_name": source_name, "rallies": rallies})
    return jsonify(result)


@app.route('/api/extract/<path:source_name>', methods=['POST'])
def extract_one(source_name):
    extracted = extract_next_rally(source_name)
    return jsonify({"extracted": extracted})


@app.route('/api/video/<path:source_name>/<path:folder_name>')
def serve_video(source_name, folder_name):
    file_path = os.path.join(WORK_FOLDER, source_name, folder_name, "raw_clip.mp4")
    return send_file(file_path, conditional=True)


@app.route('/api/json/<path:source_name>/<path:folder_name>')
def get_json(source_name, folder_name):
    json_path = os.path.join(WORK_FOLDER, source_name, folder_name, "annotations.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                return jsonify({"annotations": json.load(f)})
            except Exception:
                pass
    return jsonify({"annotations": []})


@app.route('/api/save/<path:source_name>/<path:folder_name>', methods=['POST'])
def save_json(source_name, folder_name):
    json_path = os.path.join(WORK_FOLDER, source_name, folder_name, "annotations.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(request.json, f, ensure_ascii=False, indent=4)
    return jsonify({"status": "success"})


@app.route('/api/delete/<path:source_name>/<path:folder_name>', methods=['DELETE'])
def delete_clip(source_name, folder_name):
    clip_dir = os.path.join(WORK_FOLDER, source_name, folder_name)
    if os.path.exists(clip_dir):
        import shutil as _shutil
        _shutil.rmtree(clip_dir)
    mark_deleted(source_name, folder_name)
    next_rally = extract_next_rally(source_name)
    return jsonify({"next_extracted": next_rally})


if __name__ == '__main__':
    print("服务已启动。请在浏览器中访问: https://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context='adhoc')
