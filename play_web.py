"""
Web-based HeXO game — play against the AI in your browser.

Usage:
    python play_web.py --model checkpoints_nn/model_final.pt
    python play_web.py --model checkpoints_nn/model_final.pt --num-searches 200
"""

import argparse
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hexo.game import HeXOGame, HeXOState
from alphazero.model import ResNet
from alphazero.mcts import MCTS

# Global game state
game = None
mcts = None
state = None
game_lock = threading.Lock()


def get_board_data():
    """Extract board state for the frontend."""
    cells = []
    for (q, r) in state.stones_p1:
        cells.append({'q': q, 'r': r, 'player': 1})
    for (q, r) in state.stones_p2:
        cells.append({'q': q, 'r': r, 'player': -1})

    # Get valid moves as axial coords
    if not state.all_stones:
        valid_moves = [{'q': 0, 'r': 0}]
    else:
        valid = game.get_valid_moves(state)
        valid_moves = []
        legal = np.where(valid > 0)[0]
        for action in legal:
            q, r = game.action_to_axial(state, action)
            valid_moves.append({'q': int(q), 'r': int(r)})

    last_move = None
    if state.move_history:
        lm = state.move_history[-1]
        last_move = {'q': lm[0], 'r': lm[1]}

    return {
        'cells': cells,
        'valid_moves': valid_moves,
        'current_player': state.current_player,
        'moves_left': state.moves_left_in_turn,
        'winner': state.winner,
        'move_number': state.total_moves,
        'last_move': last_move,
    }


def do_ai_move():
    """Have the AI make a move. Returns the move made."""
    global state
    action_probs = mcts.search(state)
    action = int(np.argmax(action_probs))
    q, r = game.action_to_axial(state, action)
    state = game.get_next_state(state, action, state.current_player)
    return {'q': q, 'r': r}


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>HeXO - Play vs AI</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; overflow: hidden; }
#topbar { position: fixed; top: 0; left: 0; right: 0; z-index: 10; background: rgba(26,26,46,0.95);
          padding: 8px 16px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #333; }
#topbar h1 { font-size: 22px; color: #e94560; }
#status { font-size: 16px; flex: 1; }
#thinking { display: none; color: #e94560; font-size: 14px; }
button { background: #e94560; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
button:hover { background: #c73650; }
#hint { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); color: #555; font-size: 12px; z-index: 10; }
canvas { display: block; cursor: grab; }
canvas.grabbing { cursor: grabbing; }
</style>
</head>
<body>
<div id="topbar">
    <h1>HeXO</h1>
    <div id="status">Your turn (Blue) - click a hex</div>
    <div id="thinking">AI thinking...</div>
    <button onclick="newGame()">New Game</button>
    <button onclick="undoMove()">Undo</button>
    <button onclick="resetView()">Reset View</button>
</div>
<canvas id="board"></canvas>
<div id="hint">Drag to pan | Scroll to zoom | Click highlighted hex to place</div>

<script>
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const SQRT3 = Math.sqrt(3);
const BASE_HEX_SIZE = 28;

let boardData = null;
let hoveredHex = null;

// Camera state
let camX = 0, camY = 0;  // pan offset in screen pixels
let zoom = 1.0;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let dragStartCamX = 0, dragStartCamY = 0;
let dragMoved = false;

function hexSize() { return BASE_HEX_SIZE * zoom; }

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    render();
}
window.addEventListener('resize', resizeCanvas);

function hexToScreen(q, r) {
    const sz = hexSize();
    const x = sz * (SQRT3 * q + SQRT3/2 * r) + canvas.width/2 + camX;
    const y = sz * (3/2 * r) + canvas.height/2 + camY;
    return [x, y];
}

function screenToHex(sx, sy) {
    const sz = hexSize();
    const px = (sx - canvas.width/2 - camX) / sz;
    const py = (sy - canvas.height/2 - camY) / sz;
    const q = (SQRT3/3 * px - 1/3 * py);
    const r = (2/3 * py);
    let s = -q - r;
    let rq = Math.round(q), rr = Math.round(r), rs = Math.round(s);
    const dq = Math.abs(rq - q), dr = Math.abs(rr - r), ds = Math.abs(rs - s);
    if (dq > dr && dq > ds) rq = -rr - rs;
    else if (dr > ds) rr = -rq - rs;
    return [rq, rr];
}

function drawHex(cx, cy, size, fill, stroke, lineWidth) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 180 * (60 * i - 30);
        ctx.lineTo(cx + size * Math.cos(angle), cy + size * Math.sin(angle));
    }
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth || 1; ctx.stroke(); }
}

function render() {
    if (!boardData) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const sz = hexSize();
    const isMyTurn = boardData.current_player === 1 && boardData.winner === 0;

    // Draw valid move highlights
    boardData.valid_moves.forEach(v => {
        const [sx, sy] = hexToScreen(v.q, v.r);
        if (sx < -sz || sx > canvas.width+sz || sy < -sz || sy > canvas.height+sz) return;
        const isHovered = hoveredHex && hoveredHex[0] === v.q && hoveredHex[1] === v.r;
        const fill = isHovered && isMyTurn ? 'rgba(100, 180, 255, 0.45)' : 'rgba(100, 180, 255, 0.12)';
        drawHex(sx, sy, sz - 2, fill, 'rgba(100, 180, 255, 0.25)', 1);
    });

    // Draw placed pieces
    boardData.cells.forEach(c => {
        const [sx, sy] = hexToScreen(c.q, c.r);
        if (sx < -sz || sx > canvas.width+sz || sy < -sz || sy > canvas.height+sz) return;
        const isLast = boardData.last_move && boardData.last_move.q === c.q && boardData.last_move.r === c.r;
        let fill, stroke;
        if (c.player === 1) {
            fill = '#4488ff'; stroke = isLast ? '#ffffff' : '#2266cc';
        } else {
            fill = '#e94560'; stroke = isLast ? '#ffffff' : '#c73650';
        }
        drawHex(sx, sy, sz - 2, fill, stroke, isLast ? 3 : 1.5);
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${Math.max(9, sz * 0.4)|0}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(c.q + ',' + c.r, sx, sy);
    });

    // Status
    const el = document.getElementById('status');
    if (boardData.winner !== 0) {
        const w = boardData.winner === 1 ? 'You win!' : 'AI wins!';
        el.textContent = w + ' (' + boardData.move_number + ' moves)';
        el.style.color = boardData.winner === 1 ? '#4488ff' : '#e94560';
    } else if (boardData.current_player === 1) {
        el.textContent = 'Your turn (Blue) - ' + boardData.moves_left + ' move(s) left';
        el.style.color = '#4488ff';
    } else {
        el.textContent = 'AI thinking...';
        el.style.color = '#e94560';
    }
}

// --- Pan ---
canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    dragMoved = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartCamX = camX;
    dragStartCamY = camY;
    canvas.classList.add('grabbing');
});

canvas.addEventListener('mousemove', (e) => {
    if (isDragging) {
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
        camX = dragStartCamX + dx;
        camY = dragStartCamY + dy;
        render();
    }
    // Hover
    const [q, r] = screenToHex(e.clientX, e.clientY);
    hoveredHex = [q, r];
    if (!isDragging) render();
});

canvas.addEventListener('mouseup', (e) => {
    isDragging = false;
    canvas.classList.remove('grabbing');
});

canvas.addEventListener('mouseleave', () => {
    isDragging = false;
    canvas.classList.remove('grabbing');
});

// --- Zoom ---
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const oldZoom = zoom;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    zoom = Math.max(0.2, Math.min(4, zoom * factor));
    // Zoom toward mouse position
    const mx = e.clientX - canvas.width/2;
    const my = e.clientY - canvas.height/2;
    camX = mx - (mx - camX) * (zoom / oldZoom);
    camY = my - (my - camY) * (zoom / oldZoom);
    render();
}, { passive: false });

// --- Click to place ---
canvas.addEventListener('click', async (e) => {
    if (dragMoved) return;  // Was a drag, not a click
    if (!boardData || boardData.winner !== 0 || boardData.current_player !== 1) return;

    const [q, r] = screenToHex(e.clientX, e.clientY);
    const isValid = boardData.valid_moves.some(v => v.q === q && v.r === r);
    if (!isValid) return;

    const resp = await fetch('/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({q, r})
    });
    boardData = await resp.json();
    render();

    // Trigger AI moves
    while (boardData.current_player === -1 && boardData.winner === 0) {
        document.getElementById('thinking').style.display = 'inline';
        const aiResp = await fetch('/ai_move', {method: 'POST'});
        boardData = await aiResp.json();
        document.getElementById('thinking').style.display = 'none';
        render();
    }
});

function resetView() { camX = 0; camY = 0; zoom = 1.0; render(); }

async function newGame() {
    const resp = await fetch('/new_game', {method: 'POST'});
    boardData = await resp.json();
    camX = 0; camY = 0; zoom = 1.0;
    render();
}

async function undoMove() {
    const resp = await fetch('/undo', {method: 'POST'});
    boardData = await resp.json();
    render();
}

// Init
resizeCanvas();
(async () => {
    const resp = await fetch('/state');
    boardData = await resp.json();
    render();
})();
</script>
</body>
</html>"""


class GameHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logs

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html_response(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        if self.path == '/':
            self._html_response(HTML_PAGE)
        elif self.path == '/state':
            with game_lock:
                self._json_response(get_board_data())
        else:
            self.send_error(404)

    def do_POST(self):
        global state

        if self.path == '/move':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            q, r = body['q'], body['r']

            with game_lock:
                if state.winner != 0:
                    self._json_response(get_board_data())
                    return

                if not game.is_valid_placement(state, q, r):
                    self._json_response(get_board_data())
                    return

                action = game.axial_to_action(state, q, r)
                if action == -1:
                    self._json_response(get_board_data())
                    return

                state = game.get_next_state(state, action, state.current_player)
                self._json_response(get_board_data())

        elif self.path == '/ai_move':
            with game_lock:
                if state.winner != 0 or state.current_player != -1:
                    self._json_response(get_board_data())
                    return
                do_ai_move()
                self._json_response(get_board_data())

        elif self.path == '/new_game':
            with game_lock:
                state = game.get_initial_state()
                self._json_response(get_board_data())

        elif self.path == '/undo':
            with game_lock:
                if state.move_history:
                    # Rebuild state up to last human move
                    history = list(state.move_history)
                    # Remove AI moves and last human move
                    while history and history[-1][2] == -1:
                        history.pop()
                    if history:
                        history.pop()  # Remove last human move too

                    state = game.get_initial_state()
                    for (mq, mr, mp) in history:
                        action = game.axial_to_action(state, mq, mr)
                        state = game.get_next_state(state, action, state.current_player)

                self._json_response(get_board_data())
        else:
            self.send_error(404)


def main():
    global game, mcts, state

    parser = argparse.ArgumentParser(description="Play HeXO vs AI in browser")
    parser.add_argument('--model', type=str, default='checkpoints_nn/model_final.pt')
    parser.add_argument('--num-searches', type=int, default=800,
                        help='MCTS simulations per AI move (default: 800)')
    parser.add_argument('--time-limit', type=int, default=None,
                        help='Time limit per AI move in ms (overrides --num-searches)')
    parser.add_argument('--win-length', type=int, default=5)
    parser.add_argument('--max-dist', type=int, default=8)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    game = HeXOGame(win_length=args.win_length, max_placement_dist=args.max_dist)
    state = game.get_initial_state()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model}...")
    model = ResNet(game, num_resBlocks=10, num_hidden=128, device=device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded on {device}")

    mcts_args = {
        'C': 2,
        'num_searches': args.num_searches,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3,
    }
    if args.time_limit:
        mcts_args['time_limit_ms'] = args.time_limit
    mcts = MCTS(game, mcts_args, model)

    server = HTTPServer(('localhost', args.port), GameHandler)
    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    server.serve_forever()


if __name__ == '__main__':
    main()
