import random
import networkx as nx
import matplotlib.pyplot as plt

N_AGENTS = 10
ALPHA = 0.5
NOISE_RANGE = 0.5
MAX_ITER = 30

STORAGE_COUNT = N_AGENTS
ITERATION_COUNT = 0
ARITHMETIC_COUNT = 0
MESSAGE_COUNT = 0

GLOBAL_TRUE_MEAN = None


class Agent:
    def __init__(self, agent_id, initial_value, neighbors):
        self.id = agent_id
        self.x = initial_value
        self.neighbors = list(neighbors)
        self.received_values = []

    def send_to(self, recipient_id, noise=True):
        val = self.x
        if noise:
            val += random.random() - NOISE_RANGE
        global MESSAGE_COUNT
        MESSAGE_COUNT += 1
        return val

    def receive_from_neighbors(self, agents_dict):
        received = []
        for nb_id in self.neighbors:
            val = agents_dict[nb_id].send_to(self.id, noise=True)
            if random.random() > 0.1:
                received.append(val)
        return received

    def update_state(self, received_values):
        if not received_values:
            return
        avg_neighbor = sum(received_values) / len(received_values)
        global ARITHMETIC_COUNT
        ARITHMETIC_COUNT += 2
        self.x = self.x + ALPHA * (avg_neighbor - self.x)


def run_consensus():
    global ITERATION_COUNT, GLOBAL_TRUE_MEAN

    n_agents = N_AGENTS
    print(f"Создано агентов: {n_agents}")

    G = nx.Graph()
    G.add_nodes_from(range(n_agents))
    attempts = 0
    while min(dict(G.degree()).values()) < 2 and attempts < 1000:
        attempts += 1
        i, j = random.sample(range(n_agents), 2)
        if i != j and not G.has_edge(i, j):
            G.add_edge(i, j)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 4:
                to_remove = random.sample(neighbors, len(neighbors) - 4)
                for nb in to_remove:
                    G.remove_edge(node, nb)

    agents = {}
    initial_values = [round(random.uniform(0, 100), 2) for _ in range(n_agents)]
    for i in range(n_agents):
        agents[i] = Agent(i, initial_values[i], G.neighbors(i))

    print("\n[Итерация 0]")
    current_values = [agents[i].x for i in range(n_agents)]
    print(f"Значения: {[round(v, 2) for v in current_values]}")
    true_mean = sum(initial_values) / n_agents
    GLOBAL_TRUE_MEAN = true_mean
    print(f"Истинное среднее: {true_mean:.4f}")

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='plum', node_size=800,
            font_weight='bold', edge_color='gray')
    labels = {i: f"{agents[i].x:.1f}" for i in range(n_agents)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color='black')
    plt.title("Начальное состояние графа")
    plt.show(block=False)
    plt.pause(15)
    plt.close()

    for iteration in range(1, MAX_ITER + 1):
        edges_to_remove = []
        if random.random() < 0.5:
            candidate_edges = list(G.edges())
            if candidate_edges:
                k = min(random.randint(1, 2), len(candidate_edges))
                edges_to_remove = random.sample(candidate_edges, k)
                for u, v in edges_to_remove:
                    G.remove_edge(u, v)
                    agents[u].neighbors = list(G.neighbors(u))
                    agents[v].neighbors = list(G.neighbors(v))
                print(f"\n[УДАЛЕНИЕ] Удалены связи: {edges_to_remove}")

        for i in range(n_agents):
            received = agents[i].receive_from_neighbors(agents)
            agents[i].update_state(received)

        current_values = [agents[i].x for i in range(n_agents)]
        avg_now = sum(current_values) / n_agents
        err = abs(avg_now - true_mean)
        print(f"\n[Итерация {iteration}]")
        print(f"Значения: {[round(v, 2) for v in current_values]}")
        print(f"Текущее среднее: {avg_now:.4f}, Ошибка: {err:.4f}")

        if err < 0.1 and iteration > 5:
            print("→ Достигнута достаточная сходимость.")
            break

        ITERATION_COUNT += 1

    print("\n" + "="*50)
    print("Расчёт стоимости выполнения")

    cost_storage = STORAGE_COUNT * 1
    cost_iterations = ITERATION_COUNT * 1
    cost_arithmetic = ARITHMETIC_COUNT * 0.01
    cost_messages = MESSAGE_COUNT * 0.1
    cost_final = 1000

    total_cost = cost_storage + cost_iterations + cost_arithmetic + cost_messages + cost_final

    print(f"Хранение одного числа: {STORAGE_COUNT} чисел × 1 рубль = {cost_storage:.2f} руб.")
    print(f"Одна итерация алгоритма: {ITERATION_COUNT} итераций × 1 рубль = {cost_iterations:.2f} руб.")
    print(f"Одна арифметическая операция: {ARITHMETIC_COUNT} операций × 0.01 рубля = {cost_arithmetic:.2f} руб.")
    print(f"Одно сообщение между агентами: {MESSAGE_COUNT} сообщений × 0.1 рубля = {cost_messages:.2f} руб.")
    print(f"Отправка окончательного результата пользователю: 1 раз × 1000 рублей = {cost_final} руб.")
    print("-" * 50)
    print(f"Итого: {total_cost:.2f} рублей")

    reporter = random.choice(list(agents.keys()))
    final_avg = agents[reporter].x
    print(f"\n✅ Агент {reporter} отправляет результат на сервер: {final_avg:.4f}")
    print(f"Истинное среднее было: {true_mean:.4f}")
    print(f"Абсолютная ошибка: {abs(final_avg - true_mean):.4f}")


if __name__ == "__main__":
    random.seed(42)
    run_consensus()