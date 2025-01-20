# %%
import chess
import random
import math
import time

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def select_child(self):
        log_parent_visits = math.log(self.visits + 1)  # avoid log(0)
        return max(self.children, 
                  key=lambda child: child.wins / (child.visits + 1) + 
                  math.sqrt(2 * log_parent_visits / (child.visits + 1)))

    def expand(self):
        moves = list(self.board.legal_moves)
        for move in moves:
            if move not in [child.move for child in self.children]:
                new_board = self.board.copy()
                new_board.push(move)
                new_child = Node(new_board, self, move)
                self.children.append(new_child)
                return new_child
        return None

    def update(self, result):
        self.visits += 1
        self.wins += result

def evaluate_board(board):
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # 간단한 물질점수 평가 추가
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    return score / 30  # 정규화

def mcts(root, iterations=200):  # 반복 횟수 감소
    start_time = time.time()
    
    for _ in range(iterations):
        if time.time() - start_time > 1.0:  # 1초 시간 제한
            break
            
        node = root
        depth = 0
        
        # 선택
        while not node.board.is_game_over() and node.is_fully_expanded() and depth < 10:
            node = node.select_child()
            depth += 1
        
        # 확장
        if not node.board.is_game_over() and depth < 10:
            expanded = node.expand()
            if expanded:
                node = expanded
        
        # 시뮬레이션
        board = node.board.copy()
        sim_depth = 0
        while not board.is_game_over() and sim_depth < 10:  # 시뮬레이션 깊이 제한
            moves = list(board.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            board.push(move)
            sim_depth += 1
        
        # 역전파
        result = evaluate_board(board)
        while node:
            node.update(result)
            node = node.parent
    
    # 가장 많이 방문한 자식 선택
    if not root.children:
        return random.choice(list(root.board.legal_moves))
    return max(root.children, key=lambda c: c.visits).move

def play_chess():
    board = chess.Board()
    move_count = 1
    
    while not board.is_game_over():
        print(f"\n{move_count}번째 수")
        start_time = time.time()
        
        if board.turn == chess.WHITE:
            print("백의 차례")
            root = Node(board)
            move = mcts(root, iterations=200)
        else:
            print("흑의 차례")
            root = Node(board)
            move = mcts(root, iterations=200)
        
        end_time = time.time()
        print(f"계산 시간: {end_time - start_time:.2f}초")
        
        board.push(move)
        print(board)
        print("-----")
        move_count += 1
        time.sleep(0.5)  # 대기 시간 감소
    
    print("\n게임 종료!")
    if board.is_checkmate():
        print("체크메이트! 승자:", "백" if board.turn == chess.BLACK else "흑")
    else:
        print("무승부!")

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import chess
import time

# 체스 보드를 신경망 입력으로 변환하는 함수
def board_to_input(board):
    # 12개 채널 (6개 기물 타입 x 2색)
    pieces = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP,
        chess.ROOK, chess.QUEEN, chess.KING
    ]
    board_state = np.zeros((12, 8, 8), dtype=np.float32)
    
    for i, piece in enumerate(pieces):
        for color in [chess.WHITE, chess.BLACK]:
            mask = board.pieces(piece, color)
            for square in mask:
                row, col = square // 8, square % 8
                channel = i + (0 if color else 6)
                board_state[channel][row][col] = 1
                
    return torch.FloatTensor(board_state)

# DQN 신경망
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4096)  # 최대 가능한 이동 수
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN 에이전트
class ChessAgent:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ChessNet().to(device)
        self.target_model = ChessNet().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_move(self, board):
        if random.random() < self.epsilon:
            return random.choice(list(board.legal_moves))
            
        state = board_to_input(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        
        legal_moves = list(board.legal_moves)
        legal_move_scores = []
        for move in legal_moves:
            move_idx = self.move_to_index(move)
            legal_move_scores.append((move, q_values[0][move_idx].item()))
        
        return max(legal_move_scores, key=lambda x: x[1])[0]
    
    def move_to_index(self, move):
        # 체스 이동을 인덱스로 변환 (from_square * 64 + to_square)
        return move.from_square * 64 + move.to_square
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([board_to_input(s).to(self.device) for s, _, _, _, _ in batch])
        next_states = torch.stack([board_to_input(s).to(self.device) for _, _, _, s, _ in batch])
        
        current_q = self.model(states)
        next_q = self.target_model(next_states)
        
        target = current_q.clone()
        for i, (_, action, reward, _, done) in enumerate(batch):
            move_idx = self.move_to_index(action)
            if done:
                target[i][move_idx] = reward
            else:
                target[i][move_idx] = reward + self.gamma * torch.max(next_q[i])
                
        loss = nn.MSELoss()(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 학습 함수
def train_agent(agent, num_games=10):
    white_wins = 0
    black_wins = 0
    draws = 0
    
    for game in range(num_games):
        print(f"Game {game + 1}/{num_games}")
        board = chess.Board()
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # 백: DQN
                move = agent.get_move(board)
            else:
                # 흑: MCTS
                move = mcts(board)  # 이전에 정의한 mcts 함수 사용
                
            old_board = board.copy()
            board.push(move)
            
            # 보상 계산
            reward = 0
            if board.is_game_over():
                if board.is_checkmate():
                    reward = 1 if board.turn == chess.BLACK else -1
                    if board.turn == chess.BLACK:
                        black_wins += 1
                    else:
                        white_wins += 1
                else:
                    draws += 1
                    reward = 0
            
            # 경험 저장
            if board.turn == chess.BLACK:  # 백의 수가 끝난 후
                agent.remember(old_board, move, reward, board, board.is_game_over())
                agent.replay()
                
        # 매 게임마다 타겟 네트워크 업데이트
        agent.update_target_model()
        print(f"Game {game + 1} finished. White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
    
    return agent

# 저장 및 로드 함수
def save_agent(agent, path):
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, path)

def load_agent(path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    agent = ChessAgent(device)
    checkpoint = torch.load(path, map_location=device)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    return agent

# 학습된 에이전트로 게임 플레이
def play_trained_agent(agent):
    board = chess.Board()
    move_count = 1
    
    while not board.is_game_over():
        print(f"\n{move_count}번째 수")
        start_time = time.time()
        
        if board.turn == chess.WHITE:
            print("백의 차례 (DQN)")
            move = agent.get_move(board)
        else:
            print("흑의 차례 (MCTS)")
            move = mcts(board)
        
        end_time = time.time()
        print(f"계산 시간: {end_time - start_time:.2f}초")
        
        board.push(move)
        print(board)
        print("-----")
        move_count += 1
        time.sleep(0.3)
    
    print("\n게임 종료!")
    if board.is_checkmate():
        print("체크메이트! 승자:", "백" if board.turn == chess.BLACK else "흑")
    else:
        print("무승부!")

def mcts(board, iterations=300):  # 수정된 mcts 함수
    root = Node(board)  # 보드를 Node 객체로 변환
    start_time = time.time()
    
    for _ in range(iterations):
        if time.time() - start_time > 0.5:  # 시간 제한 0.5초
            break
            
        node = root
        depth = 0
        
        # 선택
        while not node.board.is_game_over() and node.is_fully_expanded() and depth < 10:
            node = node.select_child()
            depth += 1
        
        # 확장
        if not node.board.is_game_over() and depth < 10:
            expanded = node.expand()
            if expanded:
                node = expanded
        
        # 시뮬레이션
        board = node.board.copy()
        sim_depth = 0
        while not board.is_game_over() and sim_depth < 10:
            moves = list(board.legal_moves)
            if not moves:
                break
            move = random.choice(moves)
            board.push(move)
            sim_depth += 1
        
        # 역전파
        result = evaluate_board(board)
        while node:
            node.update(result if board.turn == chess.BLACK else -result)
            node = node.parent
    
    # 가장 많이 방문한 자식 선택
    if not root.children:
        return random.choice(list(root.board.legal_moves))
    return max(root.children, key=lambda c: c.visits).move

# 사용 예시:
# 에이전트 생성 및 학습
agent = ChessAgent()
trained_agent = train_agent(agent, num_games=30)

# 학습된 모델 저장
save_agent(trained_agent, 'chess_dqn.pth')

# 나중에 학습된 모델 불러와서 게임하기
loaded_agent = load_agent('chess_dqn.pth')
play_trained_agent(loaded_agent)

# %%
# 저장된 모델 불러오기
loaded_agent = load_agent('chess_dqn.pth')

# 추가 학습 진행 (5회)
trained_agent = train_agent(loaded_agent, num_games=5)

# 추가 학습된 모델 저장 (기존 파일을 덮어쓰거나 새 파일로 저장)
#save_agent(trained_agent, 'chess_dqn_additional.pth')  # 새 파일로 저장
# 또는
save_agent(trained_agent, 'chess_dqn.pth')  # 기존 파일 덮어쓰기

# 학습된 모델로 게임 진행
play_trained_agent(trained_agent)

# %%
import torch
import torch.nn as nn
import numpy as np
import chess

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # 정책 헤드 (다음 수 예측)
        self.policy_conv = nn.Conv2d(256, 32, 1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 가능한 모든 이동
        
        # 가치 헤드 (승패 예측)
        self.value_conv = nn.Conv2d(256, 32, 1)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # 정책 출력
        policy = torch.relu(self.policy_conv(x))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)
        
        # 가치 출력
        value = torch.relu(self.value_conv(x))
        value = value.view(-1, 32 * 8 * 8)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class NeuralMCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.prior = prior  # 신경망이 예측한 이동 확률
        
    def expand(self, policy):
        for move in self.board.legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            new_board = self.board.copy()
            new_board.push(move)
            self.children.append(
                NeuralMCTSNode(
                    new_board, 
                    self, 
                    move, 
                    prior=policy[move_idx].item()
                )
            )
            
    def select_child(self, c_puct=1.0):
        # AlphaGo-style UCT
        return max(self.children, 
                  key=lambda child: (child.wins / (child.visits + 1) + 
                                   c_puct * child.prior * 
                                   np.sqrt(self.visits) / (1 + child.visits)))

def neural_mcts_search(board, model, num_simulations=100, device='cuda'):
    root = NeuralMCTSNode(board)
    
    # 보드를 신경망 입력으로 변환
    board_tensor = board_to_tensor(board).to(device)
    with torch.no_grad():
        policy, value = model(board_tensor)
    
    # 루트 노드 확장
    root.expand(policy[0])
    
    # MCTS 시뮬레이션
    for _ in range(num_simulations):
        node = root
        
        # 선택
        while node.children and not node.board.is_game_over():
            node = node.select_child()
            
        # 확장 & 평가
        if not node.board.is_game_over():
            board_tensor = board_to_tensor(node.board).to(device)
            with torch.no_grad():
                policy, value = model(board_tensor)
            node.expand(policy[0])
            v = value.item()
        else:
            # 게임 종료 상태 평가
            v = 1.0 if node.board.is_checkmate() else 0.0
            v = v if node.board.turn == chess.BLACK else -v
            
        # 역전파
        while node is not None:
            node.visits += 1
            node.wins += v
            node = node.parent
            v = -v  # 흑/백 턴 전환
            
    # 가장 많이 방문한 수 선택
    return max(root.children, key=lambda c: c.visits).move

def board_to_tensor(board):
    # 12채널 (6종류 기물 x 2색상) 표현
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
             chess.ROOK, chess.QUEEN, chess.KING]
    board_state = np.zeros((12, 8, 8), dtype=np.float32)
    
    for i, piece in enumerate(pieces):
        for color in [chess.WHITE, chess.BLACK]:
            mask = board.pieces(piece, color)
            for square in mask:
                row, col = square // 8, square % 8
                channel = i + (0 if color else 6)
                board_state[channel][row][col] = 1
                
    return torch.FloatTensor(board_state).unsqueeze(0)

# 학습 함수
def train_neural_mcts(model, optimizer, num_games=10, device='cuda'):
    model.train()
    
    for game in range(num_games):
        print(f"\n시작: 게임 {game + 1}/{num_games}")
        board = chess.Board()
        game_states = []
        move_count = 0
        
        while not board.is_game_over():
            move_count += 1
            
            state_tensor = board_to_tensor(board).to(device)
            
            # MCTS 탐색 정보 출력
            print(f"\n수 선택 중... (게임 {game + 1}, {move_count}수)")
            move = neural_mcts_search(board, model, num_simulations=50, device=device)
            
            # 선택된 수와 현재 보드 상태 출력
            print(f"선택된 수: {move}")
            print(board)
            
            game_states.append((state_tensor, move))
            board.push(move)
            
            # 매 10수마다 정책과 가치 출력
            if move_count % 10 == 0:
                with torch.no_grad():
                    policy, value = model(state_tensor)
                print(f"현재 보드 가치 예측: {value.item():.3f}")
                print(f"정책 분포 최대/최소: {policy.max().item():.3f}/{policy.min().item():.3f}")
        
        print(f"\n게임 {game + 1} 종료! 총 {move_count}수 진행됨")
        print(f"최종 보드 상태:")
        print(board)

# 사용 예시:

model = ChessNet().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
train_neural_mcts(model, optimizer, num_games=10)

# 모델 저장
torch.save(model.state_dict(), 'neural_mcts_chess.pth')

# 게임 플레이
board = chess.Board()
while not board.is_game_over():
    move = neural_mcts_search(board, model)
    board.push(move)
    print(board)
    print("-----")
