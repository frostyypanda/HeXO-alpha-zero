"""Tests for play_web.py server."""
import sys, os, json, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import urllib.request
import urllib.error

def test_imports():
    """All imports work."""
    from play_web import get_board_data, do_ai_move, GameHandler
    print("PASS: imports")

def test_board_data_empty():
    """get_board_data works on initial empty state."""
    import play_web
    from hexo.game import HeXOGame
    play_web.game = HeXOGame(win_length=5, max_placement_dist=8)
    play_web.state = play_web.game.get_initial_state()
    data = play_web.get_board_data()
    assert data['cells'] == [], f"Expected empty cells, got {data['cells']}"
    assert data['current_player'] == 1
    assert data['winner'] == 0
    assert data['moves_left'] == 1  # First move is 1 placement
    assert len(data['valid_moves']) == 1  # Only (0,0) on empty board
    assert data['valid_moves'][0] == {'q': 0, 'r': 0}
    print("PASS: board_data_empty")

def test_board_data_after_move():
    """get_board_data works after a move."""
    import play_web
    from hexo.game import HeXOGame
    play_web.game = HeXOGame(win_length=5, max_placement_dist=8)
    play_web.state = play_web.game.get_initial_state()
    # Place at (0, 0)
    action = play_web.game.axial_to_action(play_web.state, 0, 0)
    play_web.state = play_web.game.get_next_state(play_web.state, action, 1)
    data = play_web.get_board_data()
    assert len(data['cells']) == 1
    assert data['cells'][0]['q'] == 0 and data['cells'][0]['r'] == 0
    assert data['cells'][0]['player'] == 1
    assert data['current_player'] == -1  # P2's turn now
    assert data['last_move'] == {'q': 0, 'r': 0}
    assert len(data['valid_moves']) > 0
    print("PASS: board_data_after_move")

def test_valid_moves_nonempty():
    """Valid moves are correctly populated after first move."""
    import play_web
    from hexo.game import HeXOGame
    play_web.game = HeXOGame(win_length=5, max_placement_dist=4)
    play_web.state = play_web.game.get_initial_state()
    action = play_web.game.axial_to_action(play_web.state, 0, 0)
    play_web.state = play_web.game.get_next_state(play_web.state, action, 1)
    data = play_web.get_board_data()
    # All valid moves should be within dist=4 of (0,0) and not (0,0) itself
    for vm in data['valid_moves']:
        q, r = vm['q'], vm['r']
        assert not (q == 0 and r == 0), "Origin should not be valid (occupied)"
    print("PASS: valid_moves_nonempty")

def test_server_starts():
    """Server starts and serves the HTML page."""
    import play_web
    from hexo.game import HeXOGame
    from http.server import HTTPServer

    play_web.game = HeXOGame(win_length=5, max_placement_dist=4)
    play_web.state = play_web.game.get_initial_state()

    server = HTTPServer(('localhost', 9999), play_web.GameHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        # Test HTML page
        resp = urllib.request.urlopen('http://localhost:9999/')
        html = resp.read().decode()
        assert 'HeXO' in html, "HTML page should contain HeXO"
        print("PASS: server serves HTML")

        # Test /state endpoint
        resp = urllib.request.urlopen('http://localhost:9999/state')
        data = json.loads(resp.read())
        assert 'cells' in data
        assert 'valid_moves' in data
        assert data['current_player'] == 1
        print("PASS: /state endpoint")

        # Test /move endpoint
        req = urllib.request.Request(
            'http://localhost:9999/move',
            data=json.dumps({'q': 0, 'r': 0}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert len(data['cells']) == 1
        assert data['current_player'] == -1
        print("PASS: /move endpoint")

        # Test /new_game endpoint
        req = urllib.request.Request('http://localhost:9999/new_game', method='POST')
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert data['cells'] == []
        assert data['current_player'] == 1
        print("PASS: /new_game endpoint")

    finally:
        server.shutdown()

if __name__ == '__main__':
    test_imports()
    test_board_data_empty()
    test_board_data_after_move()
    test_valid_moves_nonempty()
    test_server_starts()
    print("\nAll tests passed!")
