import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

# ========== 常量定义 ==========
PRIMOGEMS_PER_WISH = 160
STAR_GLITTER_PER_WISH = 5

STAR_GLITTER_PER_FIVE_STAR_WEAPON = 10
STAR_GLITTER_PER_FOUR_STAR_WEAPON = 2

STAR_GLITTER_PER_FIVE_STAR_DUPE_CHAR = 10   # 未满命重复五星角色
STAR_GLITTER_PER_FIVE_STAR_MAXED_CHAR = 25  # 满命后重复五星角色

STAR_GLITTER_PER_FOUR_STAR_DUPE_CHAR = 2    # 未满命重复四星角色
STAR_GLITTER_PER_FOUR_STAR_MAXED_CHAR = 5   # 满命后重复四星角色

# ========== 概率表定义 ==========
# --- 五星概率 ---
_CHARACTER_5STAR_PROBS = [0.006] * 73
incremental_char = [round(0.066 + i * 0.06, 6) for i in range(16)]
_CHARACTER_5STAR_PROBS.extend(incremental_char)
_CHARACTER_5STAR_PROBS.append(1.0)

_WEAPON_5STAR_PROBS = [0.007] * 62
incremental_weapon = [
    0.07595, 0.14115, 0.20320, 0.26224, 0.31844, 0.37194, 0.42288, 0.47139,
    0.51758, 0.56156, 0.60343, 0.64327, 0.68117, 0.71719, 0.75140, 0.78387, 0.81464
]
_WEAPON_5STAR_PROBS.extend(incremental_weapon)
_WEAPON_5STAR_PROBS.append(1.0)

# --- 四星概率 ---
_COMMON_4STAR_PROBS = [0.051] * 7
_COMMON_4STAR_PROBS.extend([0.1925, 0.495])
_COMMON_4STAR_PROBS.append(1.0)

@dataclass
class WishResult:
    """单次抽卡的详细结果"""
    is_up_5: bool
    item_type: str  # '5_character', '5_weapon', '4_character', '4_weapon', '3_star'
    is_up_4: bool = False
    star_glitter_gained: int = 0
    got_special_item: bool = False   # 新增：满命重复五星角色额外道具


class GenshinWishSimulator:
    def __init__(
        self,
        char_pity_5: int = 0,
        char_guarantee_5: bool = False,
        char_pity_4: int = 0,
        char_guarantee_4: bool = False,
        wpn_pity_5: int = 0,
        wpn_guarantee_5: bool = False,
        wpn_pity_4: int = 0,
        wpn_guarantee_4: bool = False,
        primogems: int = 0,
        target_up_chars: int = 1,
        target_up_weapons: int = 1,
        up_4stars_owned: List[int] = None,
        standard_5stars_owned: List[int] = None,   # 新增：8位常驻五星命座(-1~6)
    ):
        # 五星状态
        self.char_pity_5 = char_pity_5
        self.char_guarantee_5 = char_guarantee_5
        self.wpn_pity_5 = wpn_pity_5
        self.wpn_guarantee_5 = wpn_guarantee_5

        # 四星状态
        self.char_pity_4 = char_pity_4
        self.char_guarantee_4 = char_guarantee_4
        self.wpn_pity_4 = wpn_pity_4
        self.wpn_guarantee_4 = wpn_guarantee_4

        # 资源与目标
        self.primogems = primogems
        self.star_glitter = 0
        self.total_wishes = 0
        self.total_spent = 0  # in CNY
        self.wishes_from_glitter = 0

        self.up_chars_got = 0
        self.up_weapons_got = 0
        self.target_up_chars = target_up_chars
        self.target_up_weapons = target_up_weapons

        # 新增：特殊道具计数（满命后再抽到五星角色）
        self.special_items_got = 0

        # UP四星拥有状态（三个UP四星）
        self.up_4stars_owned = up_4stars_owned if up_4stars_owned else [-1, -1, -1]

        # 新增：常驻五星8个角色命座状态
        if standard_5stars_owned is None:
            self.standard_5stars_owned = [-1] * 8
        else:
            if len(standard_5stars_owned) != 8:
                raise ValueError("standard_5stars_owned must have length 8")
            self.standard_5stars_owned = standard_5stars_owned

        # 用于日志记录
        self._char_wish_count = 0
        self._wpn_wish_count = 0

    def _handle_star_glitter(self, star_glitter: int):
        """处理获得的星辉，立即兑换为抽数（5星辉=1抽=160原石）"""
        self.star_glitter += star_glitter
        if self.star_glitter >= STAR_GLITTER_PER_WISH:
            wishes_to_add = self.star_glitter // STAR_GLITTER_PER_WISH
            self.wishes_from_glitter += wishes_to_add
            self.primogems += wishes_to_add * PRIMOGEMS_PER_WISH
            self.star_glitter %= STAR_GLITTER_PER_WISH

    # ========== 角色池单抽 ==========
    def _simulate_character_wish_once(self) -> WishResult:
        result = WishResult(is_up_5=False, item_type='3_star', star_glitter_gained=0)

        # ---- 独立判断五星 ----
        prob_5star = _CHARACTER_5STAR_PROBS[self.char_pity_5]
        is_5star = random.random() < prob_5star
        self.char_pity_5 += 1

        if is_5star:
            self.char_pity_5 = 0

            # 判定UP/歪
            if self.char_guarantee_5:
                result.is_up_5 = True
                self.char_guarantee_5 = False
            else:
                if random.random() < 0.5:
                    result.is_up_5 = True
                else:
                    self.char_guarantee_5 = True

            result.item_type = '5_character'

            # ---- 星辉逻辑：五星角色（新增常驻命座判断）----
            if result.is_up_5:
                # UP五星：沿用你原逻辑：首次获得不加，之后重复 +10
                if self.up_chars_got >= 1:
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_DUPE_CHAR
            else:
                # 常驻五星：从8个常驻里随机一个，并按命座状态决定星辉
                idx = random.randint(0, 7)
                owned = self.standard_5stars_owned[idx]
                if owned == -1:
                    # 第一次获得：标记为0命，不加星辉
                    self.standard_5stars_owned[idx] = 0
                elif owned < 6:
                    # 未满命重复：命座+1，+10星辉
                    self.standard_5stars_owned[idx] = owned + 1
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_DUPE_CHAR
                else:
                    # 已满命重复：+25星辉 + 特殊道具
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_MAXED_CHAR
                    result.got_special_item = True
                    self.special_items_got += 1

            return result  # 五星已决定，直接返回（四星不判）

        # ---- 独立判断四星（仅在未出五星时）----
        prob_4star = _COMMON_4STAR_PROBS[self.char_pity_4]
        is_4star = random.random() < prob_4star
        self.char_pity_4 += 1

        if is_4star:
            self.char_pity_4 = 0

            # 判定UP/歪
            if self.char_guarantee_4:
                is_up_4star = True
                self.char_guarantee_4 = False
            else:
                if random.random() < 0.5:
                    is_up_4star = True
                else:
                    is_up_4star = False
                    self.char_guarantee_4 = True

            result.item_type = '4_character'
            result.is_up_4 = is_up_4star

            if is_up_4star:
                chosen_idx = random.randint(0, 2)
                if self.up_4stars_owned[chosen_idx] == -1:
                    self.up_4stars_owned[chosen_idx] = 0  # 首次获得
                else:
                    self.up_4stars_owned[chosen_idx] = min(self.up_4stars_owned[chosen_idx] + 1, 6)
                    if self.up_4stars_owned[chosen_idx] == 6:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_MAXED_CHAR
                    else:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_DUPE_CHAR
            else:
                # 非UP四星角色：50%满命简化假设
                if random.random() < 0.5:
                    result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_MAXED_CHAR
                else:
                    result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_DUPE_CHAR

        return result

    # ========== 武器池单抽 ==========
    def _simulate_weapon_wish_once(self) -> WishResult:
        result = WishResult(is_up_5=False, item_type='3_star', star_glitter_gained=0)

        # ---- 独立判断五星 ----
        prob_5star = _WEAPON_5STAR_PROBS[self.wpn_pity_5]
        is_5star = random.random() < prob_5star
        self.wpn_pity_5 += 1

        if is_5star:
            self.wpn_pity_5 = 0

            if self.wpn_guarantee_5:
                result.is_up_5 = True
                self.wpn_guarantee_5 = False
            else:
                # 保持你原来的“自定义/简化”逻辑
                if random.random() < 0.75:
                    if random.random() < 0.5:
                        result.is_up_5 = True
                    else:
                        self.wpn_guarantee_5 = True
                else:
                    self.wpn_guarantee_5 = True

            result.item_type = '5_weapon'
            result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_WEAPON
            return result

        # ---- 四星（仅在未出五星时）----
        prob_4star = _COMMON_4STAR_PROBS[self.wpn_pity_4]
        is_4star = random.random() < prob_4star
        self.wpn_pity_4 += 1

        if is_4star:
            self.wpn_pity_4 = 0

            # 判定UP/歪（武器池四星UP 75%）
            if self.wpn_guarantee_4:
                is_up_4star_weapon = True
                self.wpn_guarantee_4 = False
            else:
                if random.random() < 0.75:
                    is_up_4star_weapon = True
                else:
                    is_up_4star_weapon = False
                    self.wpn_guarantee_4 = True

            if is_up_4star_weapon:
                result.item_type = '4_weapon'
                result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_WEAPON
            else:
                # 非UP四星：角色/武器按2:1分配（保留你原逻辑）
                if random.random() < (2 / 3):
                    result.item_type = '4_character'
                    if random.random() < 0.5:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_MAXED_CHAR
                    else:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_DUPE_CHAR
                else:
                    result.item_type = '4_weapon'
                    result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_WEAPON

        return result

    def _make_a_wish(self, pool: str, pulls_since_last_5: int) -> Tuple[WishResult, int, int, int]:
        """
        执行一次抽卡，返回：
        - result
        - current_pull_in_pool
        - pulls_since_last_5_before_this_pull（用于日志）
        - pulls_since_last_5_after_this_pull（用于状态更新）
        """
        self.total_wishes += 1
        self.primogems -= PRIMOGEMS_PER_WISH

        pulls_since_last_5_before = pulls_since_last_5

        if pool == 'character':
            self._char_wish_count += 1
            current_pull_in_pool = self._char_wish_count
            result = self._simulate_character_wish_once()
        else:
            self._wpn_wish_count += 1
            current_pull_in_pool = self._wpn_wish_count
            result = self._simulate_weapon_wish_once()

        # 更新 pulls_since_last_5（抽到五星 -> 归零，否则 +1）
        if result.item_type.startswith('5_'):
            pulls_since_last_5_after = 0
        else:
            pulls_since_last_5_after = pulls_since_last_5 + 1

        return result, current_pull_in_pool, pulls_since_last_5_before, pulls_since_last_5_after

    def run_single_simulation(self) -> Dict[str, Any]:
        """运行单次详细模拟（只记录4/5星日志）"""
        detailed_log = []
        pulls_since_last_5 = 0

        while (self.up_chars_got < self.target_up_chars) or (self.up_weapons_got < self.target_up_weapons):
            # 原石不足 -> 氪金
            if self.primogems < PRIMOGEMS_PER_WISH:
                self.total_spent += 648
                self.primogems += 8080
                detailed_log.append(f"【氪金】花费648元，获得8080原石。当前总花费: {self.total_spent}元")

            # 选池策略：优先未达成的目标
            pool_to_pull = 'character' if (self.up_chars_got < self.target_up_chars) else 'weapon'
            if self.up_weapons_got >= self.target_up_weapons:
                pool_to_pull = 'character'
            if self.up_chars_got >= self.target_up_chars:
                pool_to_pull = 'weapon'

            result, current_pull, pulls_before, pulls_after = self._make_a_wish(pool_to_pull, pulls_since_last_5)
            pulls_since_last_5 = pulls_after

            # 只在4/5星时记录日志
            if result.item_type.startswith('5_'):
                pool_name = "角色" if pool_to_pull == 'character' else "武器"
                log_parts = [f"【{pool_name}池第{current_pull}抽】距上个5星{pulls_before}抽"]

                if result.is_up_5:
                    if pool_to_pull == 'character':
                        self.up_chars_got += 1
                        log_parts.append(f"获得五星UP角色! ({self.up_chars_got}/{self.target_up_chars})")
                    else:
                        self.up_weapons_got += 1
                        log_parts.append(f"获得五星UP武器! ({self.up_weapons_got}/{self.target_up_weapons})")
                else:
                    if pool_to_pull == 'character':
                        log_parts.append("获得五星常驻角色")
                        if result.got_special_item:
                            log_parts.append("（满命重复：+25星辉，并获得特殊道具）")
                    else:
                        log_parts.append("获得五星常驻/非定轨武器")

                if result.star_glitter_gained > 0:
                    log_parts.append(f"获得{result.star_glitter_gained}星辉")

                detailed_log.append(" ".join(log_parts))

            elif result.item_type.startswith('4_'):
                pool_name = "角色" if pool_to_pull == 'character' else "武器"
                item_name = "角色" if 'character' in result.item_type else "武器"
                up_status = "UP" if result.is_up_4 else "非UP"
                log_parts = [f"【{pool_name}池第{current_pull}抽】获得{up_status}四星{item_name}"]
                if result.star_glitter_gained > 0:
                    log_parts.append(f"获得{result.star_glitter_gained}星辉")
                detailed_log.append(" ".join(log_parts))

            # 星辉兑换
            if result.star_glitter_gained > 0:
                self._handle_star_glitter(result.star_glitter_gained)

            # 目标达成检查
            if (self.up_chars_got >= self.target_up_chars) and (self.up_weapons_got >= self.target_up_weapons):
                break

        return {
            'log': detailed_log,
            'total_wishes': self.total_wishes,
            'total_spent': self.total_spent,
            'wishes_from_glitter': self.wishes_from_glitter,
            'final_up_4stars': self.up_4stars_owned,
            'final_standard_5stars': self.standard_5stars_owned,
            'special_items_got': self.special_items_got,
        }

    def run_multiple_simulations(self, n: int) -> Dict[str, Any]:
        """运行多次模拟并统计结果"""
        all_wishes = []
        all_spent = []
        all_glitter_wishes = []
        all_special_items = []

        for _ in range(n):
            simulator = GenshinWishSimulator(
                char_pity_5=self.char_pity_5,
                char_guarantee_5=self.char_guarantee_5,
                char_pity_4=self.char_pity_4,
                char_guarantee_4=self.char_guarantee_4,
                wpn_pity_5=self.wpn_pity_5,
                wpn_guarantee_5=self.wpn_guarantee_5,
                wpn_pity_4=self.wpn_pity_4,
                wpn_guarantee_4=self.wpn_guarantee_4,
                primogems=self.primogems,
                target_up_chars=self.target_up_chars,
                target_up_weapons=self.target_up_weapons,
                up_4stars_owned=self.up_4stars_owned.copy(),
                standard_5stars_owned=self.standard_5stars_owned.copy(),
            )
            result = simulator.run_single_simulation()
            all_wishes.append(result['total_wishes'])
            all_spent.append(result['total_spent'])
            all_glitter_wishes.append(result['wishes_from_glitter'])
            all_special_items.append(result['special_items_got'])

        wishes_array = np.array(all_wishes)
        spent_array = np.array(all_spent)
        glitter_array = np.array(all_glitter_wishes)
        special_array = np.array(all_special_items)

        def get_stats(arr):
            return {
                'mean': float(np.mean(arr)),
                'var': float(np.var(arr)),
                'std': float(np.std(arr)),
                'percentiles': {
                    10: float(np.percentile(arr, 10)),
                    25: float(np.percentile(arr, 25)),
                    50: float(np.percentile(arr, 50)),
                    75: float(np.percentile(arr, 75)),
                    99: float(np.percentile(arr, 99)),
                }
            }

        # 绘制分布图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(wishes_array, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Total Wishes Distribution')
        plt.xlabel('Number of Wishes')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(spent_array, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Total Spent (CNY) Distribution')
        plt.xlabel('Amount Spent (CNY)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('genshin_wish_simulation_results.png')
        plt.close()

        return {
            'wishes_stats': get_stats(wishes_array),
            'spent_stats': get_stats(spent_array),
            'avg_glitter_wishes': float(np.mean(glitter_array)),
            'special_items_stats': get_stats(special_array),
        }


def main():
    # ========== 配置初始状态 ==========
    char_pity_5 = 77
    char_guarantee_5 = True
    char_pity_4 = 0
    char_guarantee_4 = False

    wpn_pity_5 = 38
    wpn_guarantee_5 = True
    wpn_pity_4 = 0
    wpn_guarantee_4 = False

    primogems = 50000
    target_up_chars = 7
    target_up_weapons = 1

    up_4stars_owned = [4, 6, 6]

    # 8位常驻五星命座状态（-1~6）
    # 示例：假设你已经有其中1个满命、1个4命、其余未拥有
    standard_5stars_owned = [1, 2, -1, 6, 5, 3, 3, 4]

    num_runs = 50000

    simulator = GenshinWishSimulator(
        char_pity_5=char_pity_5,
        char_guarantee_5=char_guarantee_5,
        char_pity_4=char_pity_4,
        char_guarantee_4=char_guarantee_4,
        wpn_pity_5=wpn_pity_5,
        wpn_guarantee_5=wpn_guarantee_5,
        wpn_pity_4=wpn_pity_4,
        wpn_guarantee_4=wpn_guarantee_4,
        primogems=primogems,
        target_up_chars=target_up_chars,
        target_up_weapons=target_up_weapons,
        up_4stars_owned=up_4stars_owned,
        standard_5stars_owned=standard_5stars_owned,
    )

    if num_runs == 1:
        result = simulator.run_single_simulation()
        print("===== 单次模拟日志（仅4/5星） =====")
        for line in result['log']:
            print(line)

        print("\n===== 最终结果 =====")
        print(f"总抽数: {result['total_wishes']}")
        print(f"总花费: {result['total_spent']} 元")
        print(f"星辉兑换抽数: {result['wishes_from_glitter']}")
        print(f"最终UP四星命座: {result['final_up_4stars']}")
        print(f"最终常驻五星命座: {result['final_standard_5stars']}")
        print(f"满命重复获得的特殊道具数量: {result['special_items_got']}")
    else:
        stats = simulator.run_multiple_simulations(num_runs)
        print("===== 多次模拟统计结果 =====")

        print("抽数统计:")
        print(f"  均值: {stats['wishes_stats']['mean']:.2f}")
        print(f"  方差: {stats['wishes_stats']['var']:.2f}")
        print(f"  标准差: {stats['wishes_stats']['std']:.2f}")
        print(f"  10%分位数: {stats['wishes_stats']['percentiles'][10]:.2f}")
        print(f"  25%分位数: {stats['wishes_stats']['percentiles'][25]:.2f}")
        print(f"  50%分位数 (中位数): {stats['wishes_stats']['percentiles'][50]:.2f}")
        print(f"  75%分位数: {stats['wishes_stats']['percentiles'][75]:.2f}")
        print(f"  99%分位数: {stats['wishes_stats']['percentiles'][99]:.2f}")

        print("\n花费统计 (CNY):")
        print(f"  均值: {stats['spent_stats']['mean']:.2f}")
        print(f"  方差: {stats['spent_stats']['var']:.2f}")
        print(f"  标准差: {stats['spent_stats']['std']:.2f}")
        print(f"  10%分位数: {stats['spent_stats']['percentiles'][10]:.2f}")
        print(f"  25%分位数: {stats['spent_stats']['percentiles'][25]:.2f}")
        print(f"  50%分位数 (中位数): {stats['spent_stats']['percentiles'][50]:.2f}")
        print(f"  75%分位数: {stats['spent_stats']['percentiles'][75]:.2f}")
        print(f"  99%分位数: {stats['spent_stats']['percentiles'][99]:.2f}")

        print(f"\n平均通过星辉兑换的抽数: {stats['avg_glitter_wishes']:.2f}")

        print("\n特殊道具统计:")
        print(f"  均值: {stats['special_items_stats']['mean']:.4f}")
        print(f"  方差: {stats['special_items_stats']['var']:.4f}")
        print(f"  标准差: {stats['special_items_stats']['std']:.4f}")
        print(f"  10%分位数: {stats['special_items_stats']['percentiles'][10]:.4f}")
        print(f"  25%分位数: {stats['special_items_stats']['percentiles'][25]:.4f}")
        print(f"  50%分位数 (中位数): {stats['special_items_stats']['percentiles'][50]:.4f}")
        print(f"  75%分位数: {stats['special_items_stats']['percentiles'][75]:.4f}")
        print(f"  99%分位数: {stats['special_items_stats']['percentiles'][99]:.4f}")

        print("\n概率分布图已保存为 'genshin_wish_simulation_results.png'")

if __name__ == "__main__":
    main()