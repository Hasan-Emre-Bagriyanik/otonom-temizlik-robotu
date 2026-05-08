import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# Grid boyutunu 5x5 olarak seçtim. Daha büyük yapsaydım Q-Table çok şişecekti.
GRID_SIZE = 5
MAX_BATTERY = 100
MAX_STEPS = 200

# Şarj istasyonunu sol üst köşede sabit tutuyorum.
CHARGE_STATION = (0, 0)
# Mağazada iki tane raf var, robot bu hücrelere giremiyor.
OBSTACLES = {(1, 2), (3, 1)}

# Her bölümde aynı başlangıç durumu olsun diye kirlilik haritasını sabit yaptım.
INITIAL_DIRT = {
    (2, 2): 2,
    (4, 4): 2,
    (0, 3): 1,
    (1, 4): 1,
    (3, 3): 1,
    (4, 0): 1,
}

# Aksiyon idlerini sabit olarak tanımladım, kodun içinde rakam yerine isim yazıyorum.
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CLEAN = 4
ACTION_CHARGE = 5

# Hareket aksiyonlarının x ve y eksenindeki etkilerini bir tabloda topladım.
ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Render başlığında gösterilecek aksiyon isimleri burada tutuluyor.
ACTION_NAMES = {
    ACTION_UP: "Yukari",
    ACTION_DOWN: "Asagi",
    ACTION_LEFT: "Sol",
    ACTION_RIGHT: "Sag",
    ACTION_CLEAN: "Temizle",
    ACTION_CHARGE: "Sarja Git",
}

# Her hücrenin kirlilik seviyesine göre renk atadım.
CELL_COLORS = {
    0: "#90EE90",
    1: "#FFD700",
    2: "#FF6347",
}
OBSTACLE_COLOR = "#404040"


class StoreCleaningEnv:

    def __init__(self, seed: int = 42):
        # Seed sabit olunca her çalıştırmada aynı sonuç çıkıyor.
        self.rng = np.random.default_rng(seed)

        self.robot_pos = (0, 0)
        self.battery = MAX_BATTERY
        self.dirt_map = dict(INITIAL_DIRT)
        self.step_count = 0
        self.last_action = None
        self.total_reward = 0.0
        self.reset()

    def reset(self):
        # Robot her bölümün başında şarj istasyonundan başlasın istiyorum.
        self.robot_pos = (0, 0)
        self.battery = MAX_BATTERY
        # dict() ile kopya alıyorum, yoksa orijinal sözlük bozuluyor.
        self.dirt_map = dict(INITIAL_DIRT)
        self.step_count = 0
        self.last_action = None
        self.total_reward = 0.0
        return self._get_state()

    def step(self, action: int):
        self.last_action = action
        reward = 0.0
        done = False
        info = {}

        # Hareket aksiyonlarında önce yeni konumu hesaplıyorum.
        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            dx, dy = ACTION_DELTAS[action]
            new_x = self.robot_pos[0] + dx
            new_y = self.robot_pos[1] + dy

            # Grid dışına çıkma veya engele girme durumlarını kontrol ediyorum.
            if (
                new_x < 0
                or new_x >= GRID_SIZE
                or new_y < 0
                or new_y >= GRID_SIZE
                or (new_x, new_y) in OBSTACLES
            ):
                reward = -5
            else:
                self.robot_pos = (new_x, new_y)
                # Her adımın küçük bir maliyeti olsun ki ajan gereksiz dolaşmasın.
                reward = -1
            self.battery -= 1

        # Temizleme aksiyonunda hücrenin kirlilik seviyesine bakıyorum.
        elif action == ACTION_CLEAN:
            dirt_level = self.dirt_map.get(self.robot_pos, 0)
            if dirt_level == 2:
                # Çok kirli hücreye yüksek ödül veriyorum ki ajan oraya yönelsin.
                reward = 20
                self.dirt_map[self.robot_pos] = 0
            elif dirt_level == 1:
                reward = 10
                self.dirt_map[self.robot_pos] = 0
            else:
                # Zaten temiz olan hücreyi temizlemeye çalışmak israf, küçük ceza veriyorum.
                reward = -2
            self.battery -= 2

        # Şarja gitme aksiyonunda istasyon kontrolü yapıyorum.
        elif action == ACTION_CHARGE:
            if self.robot_pos == CHARGE_STATION:
                # Sadece pil yarısının altına düştüyse şarj mantıklı oluyor.
                # Bu kontrolü eklemeden önce ajan istasyonda kalıp dur dur şarj yapıyordu.
                if self.battery <= 50:
                    self.battery = MAX_BATTERY
                    reward = 5
                else:
                    reward = -2
            else:
                # İstasyon dışında şarja git demek anlamsız olduğu için ceza koyuyorum.
                reward = -1
            self.battery -= 1

        # Pil eksiyse 0'a sabitliyorum, negatif göstermek mantıklı olmazdı.
        if self.battery < 0:
            self.battery = 0

        # Bölüm bitiş kontrolleri burada yapılıyor.
        # Tüm kirli hücreler temizlendiyse bölüm başarıyla bitiyor.
        if all(level == 0 for level in self.dirt_map.values()):
            reward += 100
            done = True
            info["result"] = "success"
        # Pil bittiyse ve robot istasyonda değilse bölüm başarısızla bitiyor.
        elif self.battery <= 0 and self.robot_pos != CHARGE_STATION:
            reward = -100
            done = True
            info["result"] = "battery_dead"
        # Maksimum adım sayısına ulaşıldıysa bölüm timeout ile bitiyor.
        elif self.step_count + 1 >= MAX_STEPS:
            done = True
            info["result"] = "timeout"

        self.step_count += 1
        self.total_reward += reward
        return self._get_state(), reward, done, info

    def render(self):
        # Her render çağrısında yeni bir figure açıyorum, sonunda kapatmam gerekiyor.
        fig, ax = plt.subplots(figsize=(7, 7.5))

        action_name = ACTION_NAMES.get(self.last_action, "—")
        title = (
            f"Adim: {self.step_count}  |  Aksiyon: {action_name}  |  "
            f"Sarj: %{self.battery}  |  Toplam Odul: {self.total_reward:.0f}"
        )
        ax.set_title(title, fontsize=11, pad=12)

        # Grid hücrelerini tek tek dolaşarak çiziyorum.
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) in OBSTACLES:
                    color = OBSTACLE_COLOR
                else:
                    dirt_level = self.dirt_map.get((x, y), 0)
                    color = CELL_COLORS[dirt_level]

                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.2,
                )
                ax.add_patch(rect)

                # Engellerin üzerine RAF yazısı koyuyorum, anlaşılır olsun diye.
                if (x, y) in OBSTACLES:
                    ax.text(
                        x, y, "RAF",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold",
                    )

        # Şarj istasyonunu sarı yıldız olarak işaretliyorum.
        cs_x, cs_y = CHARGE_STATION
        ax.scatter(
            cs_x, cs_y,
            marker="*", s=600,
            c="gold", edgecolors="black", linewidths=1.5,
            zorder=3,
        )

        # Robotu mavi daire olarak çiziyorum, üzerine R harfi koyuyorum.
        rx, ry = self.robot_pos
        ax.scatter(
            rx, ry,
            marker="o", s=500,
            c="royalblue", edgecolors="black", linewidths=1.5,
            zorder=4,
        )
        ax.text(
            rx, ry, "R",
            ha="center", va="center",
            color="white", fontsize=12, fontweight="bold",
            zorder=5,
        )

        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_aspect("equal")
        # Y eksenini ters çeviriyorum ki standart ekran kordinatı gibi gözüksün.
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Pil seviyesini bir bar olarak alta ekliyorum.
        bar_battery = self.battery / MAX_BATTERY
        # Pil düştükçe bar rengi yeşilden kırmızıya kayıyor.
        if self.battery <= 15:
            bar_color = "#D32F2F"
        elif self.battery <= 40:
            bar_color = "#F57C00"
        elif self.battery <= 75:
            bar_color = "#FBC02D"
        else:
            bar_color = "#388E3C"

        ax.barh(
            -1.2, bar_battery * (GRID_SIZE - 1),
            height=0.25, left=-0.5,
            color=bar_color, edgecolor="black",
        )
        ax.text(
            (GRID_SIZE - 1) / 2 - 0.5, -1.55,
            f"Sarj: %{self.battery}",
            ha="center", va="center", fontsize=10,
        )

        # Lejant ekledim ki izleyen kişi renklerin ne anlama geldiğini bilsin.
        legend_y = GRID_SIZE - 0.2
        legend_items = [
            ("#90EE90", "Temiz"),
            ("#FFD700", "Kirli"),
            ("#FF6347", "Cok Kirli"),
            ("#404040", "Engel"),
        ]
        for i, (col, label) in enumerate(legend_items):
            ax.add_patch(
                Rectangle(
                    (-0.4 + i * 1.2, legend_y),
                    0.3, 0.25,
                    facecolor=col, edgecolor="black",
                )
            )
            ax.text(
                -0.05 + i * 1.2, legend_y + 0.12,
                label, fontsize=8, va="center",
            )

        ax.set_ylim(GRID_SIZE + 0.3, -1.8)

        # Figure'ı numpy array haline çeviriyorum, GIF üretirken bu şekilde lazım oluyor.
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()

        # Burayı unutursam bellek dolup kod çöküyor, mutlaka kapatmam gerekiyor.
        plt.close(fig)
        return rgb

    def _get_state(self):
        # State olarak (x, y, şarj bandı, mevcut hücre kirliliği) tuple döneriyorum.
        return (
            self.robot_pos[0],
            self.robot_pos[1],
            self._battery_band(self.battery),
            self.dirt_map.get(self.robot_pos, 0),
        )

    @staticmethod
    def _battery_band(battery: int) -> int:
        # Sürekli pil değerini 4 banda indirgiyorum çünkü Q-Table'a 100 ayrı değer sığmıyor.
        if battery <= 15:
            return 0
        if battery <= 40:
            return 1
        if battery <= 75:
            return 2
        return 3
