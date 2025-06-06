import gymnasium as gym
import torch
from LunarAgent import Agent  # Importa a classe Agent do seu arquivo original

# Configurações do ambiente
env = gym.make("LunarLander-v3", render_mode="human")  # Modo de renderização ativado
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inicializa o agente (com a mesma arquitetura usada no treino)
agent = Agent(state_size, action_size)

# Carrega os pesos do modelo treinado
checkpoint_path = "LunarLander_solved.pth"  # Substitua pelo seu arquivo .pth
agent.local_network.load_state_dict(torch.load(checkpoint_path))
agent.local_network.eval()  # Coloca o modelo em modo de avaliação (sem backpropagation)

# Testa o modelo em 10 episódios
for episode in range(10):
    state, _ = env.reset()
    score = 0
    done = False
    
    while not done:
        action = agent.act(state, epsilon=0.0)  # epsilon=0.0 → sempre usa ações ótimas
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        score += reward
    
    print(f"Episode {episode + 1} | Score: {score:.2f}")

env.close()