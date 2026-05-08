import numpy as np


class QLearningAgent:

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        # Öğrenme hızını kontrol eden parametre, klasik Q-Learning değeri olarak 0.1 verdim.
        self.alpha = alpha
        # Gelecekteki ödüllerin bugünkü değerini belirleyen indirgeme faktörü.
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # Her bölüm sonunda epsilon bu çarpanla azalıyor.
        self.epsilon_decay = epsilon_decay
        self.n_actions = 6

        # Q-Table'ı 5x5x4x3x6 boyutlu numpy array olarak tuttum, toplam 1800 hücre oluyor.
        # Sözlük yerine numpy kullanmamın sebebi indekslemenin daha hızlı olması.
        self.q_table = np.zeros((5, 5, 4, 3, 6))

        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, training: bool = True) -> int:
        # Eğitim sırasında epsilon olasılıkla rastgele aksiyon seçiyorum, bu keşif kısmı.
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Aksi durumda en yüksek Q değerli aksiyonu seçiyorum, bu da sömürü kısmı.
        q_values = self.q_table[state]
        max_q = np.max(q_values)

        # Birden fazla aksiyon aynı maksimum değere sahipse aralarından rastgele birini seçiyorum.
        # Yoksa np.argmax hep ilkini seçiyor ve simetrik durumlarda yanlı kararlar oluşuyor.
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, state, action: int, reward: float, next_state, done: bool):
        # Q-Learning'in kalbi olan Bellman denklemini buradan uyguluyorum.
        current_q = self.q_table[state + (action,)]

        # Bölüm bittiyse next_state'in Q değerini hesaba katmıyorum çünkü devam etmeyecek.
        if done:
            target = reward
        else:
            # Aksi halde bir sonraki state'in en iyi Q değerini de hesaba katıyorum.
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Mevcut değeri target'a doğru alpha kadar taşıyorum, yumuşak bir güncelleme oluyor.
        self.q_table[state + (action,)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        # Epsilon'u azaltırken minimum değerin altına düşmemesine dikkat ediyorum.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, filepath: str):
        np.save(filepath, self.q_table)

    def load(self, filepath: str):
        self.q_table = np.load(filepath)


# Karşılaştırma yapmak için tamamen rastgele aksiyon seçiyor.
class RandomAgent:

    def __init__(self, n_actions: int = 6, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, training: bool = False) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def update(self, *args, **kwargs):
        # Random ajan öğrenmiyor ama metod ismi aynı olsun ki kod bozulmasın.
        pass

    def decay_epsilon(self):
        pass
