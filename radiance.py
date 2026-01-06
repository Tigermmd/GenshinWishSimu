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
    got_special_item: bool = False   # 满命重复五星角色额外道具
    is_radiance: bool = False        # 仅角色池：捕获明光命中（来自小保底概率提升部分）


class GenshinWishSimulator:
    def __init__(
        self,
        char_pity_5: int = 0,
        char_guarantee_5: bool = False,
        wpn_pity_5: int = 0,
        wpn_guarantee_5: bool = False,
        primogems: int = 0,
        target_up_chars: int = 1,
        target_up_weapons: int = 1,
        up_4stars_owned: List[int] = None,
        standard_5stars_owned: List[int] = None,   # 8位常驻五星命座(-1~6)
        radiance_stage: int = 0,                   # 新增：明光stage(0~3)
        non_up_4star_maxed_prob: float = 0.5,      # 新增：非UP四星角色“已满命”概率
    ):
        # 五星状态
        self.char_pity_5 = char_pity_5
        self.char_guarantee_5 = char_guarantee_5
        self.wpn_pity_5 = wpn_pity_5
        self.wpn_guarantee_5 = wpn_guarantee_5

        # 四星状态（不再从初始参数传入；默认从头开始）
        self.char_pity_4 = 0
        self.char_guarantee_4 = False
        self.wpn_pity_4 = 0
        self.wpn_guarantee_4 = False

        # Radiance
        if radiance_stage not in (0, 1, 2, 3):
            raise ValueError("radiance_stage must be in {0,1,2,3}")
        self.radiance_stage = radiance_stage

        # 非UP四星满命概率
        if not (0.0 <= non_up_4star_maxed_prob <= 1.0):
            raise ValueError("non_up_4star_maxed_prob must be in [0,1]")
        self.non_up_4star_maxed_prob = non_up_4star_maxed_prob

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

        # 特殊道具计数
        self.special_items_got = 0

        # UP四星拥有状态
        self.up_4stars_owned = up_4stars_owned if up_4stars_owned else [-1, -1, -1]

        # 常驻五星命座
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
        """处理获得的星辉，立即兑换"""
        self.star_glitter += star_glitter
        if self.star_glitter >= STAR_GLITTER_PER_WISH:
            wishes_to_add = self.star_glitter // STAR_GLITTER_PER_WISH
            self.wishes_from_glitter += wishes_to_add
            self.primogems += wishes_to_add * PRIMOGEMS_PER_WISH
            self.star_glitter %= STAR_GLITTER_PER_WISH

    # ========== Radiance helper ==========
    def _radiance_up_prob(self) -> float:
        # stage0/1: 50%; stage2: 60%; stage3: 100%
        if self.radiance_stage == 2:
            return 0.60
        if self.radiance_stage == 3:
            return 1.00
        return 0.50

    def _radiance_update_on_up_small(self):
        # 小保底出UP（含Radiance） -> 回退
        if self.radiance_stage in (0, 1):
            self.radiance_stage = 0
        else:
            self.radiance_stage = 1

    def _radiance_update_on_up_big(self):
        # 大保底出UP -> stage + 1（封顶3）
        self.radiance_stage = min(self.radiance_stage + 1, 3)

    # ========== 角色池单抽 ==========
    def _simulate_character_wish_once(self) -> WishResult:
        result = WishResult(is_up_5=False, item_type='3_star', star_glitter_gained=0)

        # ---- 独立判断五星 ----
        prob_5star = _CHARACTER_5STAR_PROBS[self.char_pity_5]
        is_5star = random.random() < prob_5star
        self.char_pity_5 += 1

        if is_5star:
            self.char_pity_5 = 0
            result.item_type = '5_character'

            # UP/歪判定：大保底与Radiance共存
            if self.char_guarantee_5:
                # 大保底给UP
                result.is_up_5 = True
                self.char_guarantee_5 = False
                self._radiance_update_on_up_big()
            else:
                # 小保底：UP概率提升
                r = random.random()
                up_prob = self._radiance_up_prob()

                if r < up_prob:
                    result.is_up_5 = True
                    # 命中“增益部分”标记为Radiance
                    if self.radiance_stage >= 2 and r >= 0.5:
                        result.is_radiance = True
                    self._radiance_update_on_up_small()
                else:
                    result.is_up_5 = False
                    self.char_guarantee_5 = True

            # ---- 星辉逻辑：五星角色（含常驻8人命座 + 特殊道具）----
            if result.is_up_5:
                if self.up_chars_got >= 1:
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_DUPE_CHAR
            else:
                idx = random.randint(0, 7)
                owned = self.standard_5stars_owned[idx]
                if owned == -1:
                    self.standard_5stars_owned[idx] = 0
                elif owned < 6:
                    self.standard_5stars_owned[idx] = owned + 1
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_DUPE_CHAR
                else:
                    result.star_glitter_gained = STAR_GLITTER_PER_FIVE_STAR_MAXED_CHAR
                    result.got_special_item = True
                    self.special_items_got += 1

            return result

        # ---- 独立判断四星（仅在未出五星时）----
        prob_4star = _COMMON_4STAR_PROBS[self.char_pity_4]
        is_4star = random.random() < prob_4star
        self.char_pity_4 += 1

        if is_4star:
            self.char_pity_4 = 0

            # 角色池四星：50/50 + guarantee（从头开始）
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
                    self.up_4stars_owned[chosen_idx] = 0
                else:
                    self.up_4stars_owned[chosen_idx] = min(self.up_4stars_owned[chosen_idx] + 1, 6)
                    if self.up_4stars_owned[chosen_idx] == 6:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_MAXED_CHAR
                    else:
                        result.star_glitter_gained = STAR_GLITTER_PER_FOUR_STAR_DUPE_CHAR
            else:
                # 非UP四星角色：满命概率参数化
                if random.random() < self.non_up_4star_maxed_prob:
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
                # 保留你原来的“自定义/简化”逻辑（定轨上限=1）
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

            # 武器池四星UP 75% + guarantee（从头开始）
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
        返回：
        - result
        - current_pull_in_pool
        - pulls_since_last_5_before_this_pull（用于日志）
        - pulls_since_last_5_after_this_pull（用于状态）
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
            if self.primogems < PRIMOGEMS_PER_WISH:
                self.total_spent += 648
                self.primogems += 8080
                detailed_log.append(f"【氪金】花费648元，获得8080原石。当前总花费: {self.total_spent}元")

            pool_to_pull = 'character' if (self.up_chars_got < self.target_up_chars) else 'weapon'
            if self.up_weapons_got >= self.target_up_weapons:
                pool_to_pull = 'character'
            if self.up_chars_got >= self.target_up_chars:
                pool_to_pull = 'weapon'

            result, current_pull, pulls_before, pulls_after = self._make_a_wish(pool_to_pull, pulls_since_last_5)
            pulls_since_last_5 = pulls_after

            if result.item_type.startswith('5_'):
                pool_name = "角色" if pool_to_pull == 'character' else "武器"
                log_parts = [f"【{pool_name}池第{current_pull}抽】距上个5星{pulls_before}抽"]

                if result.is_up_5:
                    if pool_to_pull == 'character':
                        self.up_chars_got += 1
                        msg = f"获得五星UP角色! ({self.up_chars_got}/{self.target_up_chars})"
                        if result.is_radiance:
                            msg += "【Radiance】"
                        log_parts.append(msg)
                        log_parts.append(f"(radiance_stage={self.radiance_stage})")
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

            if result.star_glitter_gained > 0:
                self._handle_star_glitter(result.star_glitter_gained)

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
            'final_radiance_stage': self.radiance_stage,
        }

    def run_multiple_simulations(self, n: int) -> Dict[str, Any]:
        """运行多次模拟并统计结果（含分位数10/25/50/75/99）"""
        all_wishes = []
        all_spent = []
        all_glitter_wishes = []
        all_special_items = []

        for _ in range(n):
            simulator = GenshinWishSimulator(
                char_pity_5=self.char_pity_5,
                char_guarantee_5=self.char_guarantee_5,
                wpn_pity_5=self.wpn_pity_5,
                wpn_guarantee_5=self.wpn_guarantee_5,
                primogems=self.primogems,
                target_up_chars=self.target_up_chars,
                target_up_weapons=self.target_up_weapons,
                up_4stars_owned=self.up_4stars_owned.copy(),
                standard_5stars_owned=self.standard_5stars_owned.copy(),
                radiance_stage=self.radiance_stage,
                non_up_4star_maxed_prob=self.non_up_4star_maxed_prob,
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

        def get_stats(arr: np.ndarray) -> Dict[str, Any]:
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
    # 角色大/小保底状态
    char_pity_5 = 77
    char_guarantee_5 = True

    # 武器大/小保底状态
    wpn_pity_5 = 38
    wpn_guarantee_5 = True

    primogems = 50000 # 原石数，请将已有星辉也折算进入
    target_up_chars = 7
    target_up_weapons = 1

    up_4stars_owned = [4, 6, 6] # -1～6

    standard_5stars_owned = [1, 2, -1, 6, 5, 3, 3, 4]

    radiance_stage = 0 # 捕获明光状态，详见 radiance.jpg
    non_up_4star_maxed_prob = 0.5 # 自己估一个所有4星的满命率

    num_runs = 10000  # 1为单次模拟，>1为统计模型

    simulator = GenshinWishSimulator(
        char_pity_5=char_pity_5,
        char_guarantee_5=char_guarantee_5,
        wpn_pity_5=wpn_pity_5,
        wpn_guarantee_5=wpn_guarantee_5,
        primogems=primogems,
        target_up_chars=target_up_chars,
        target_up_weapons=target_up_weapons,
        up_4stars_owned=up_4stars_owned,
        standard_5stars_owned=standard_5stars_owned,
        radiance_stage=radiance_stage,
        non_up_4star_maxed_prob=non_up_4star_maxed_prob,
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
        print(f"最终Radiance stage: {result['final_radiance_stage']}")
    else:
        stats = simulator.run_multiple_simulations(num_runs)
        print("===== 多次模拟统计结果 =====")
        print(f"抽数统计:")
        print(f"  均值: {stats['wishes_stats']['mean']:.2f}")
        print(f"  方差: {stats['wishes_stats']['var']:.2f}")
        print(f"  10%分位数: {stats['wishes_stats']['percentiles'][10]:.2f}")
        print(f"  25%分位数: {stats['wishes_stats']['percentiles'][25]:.2f}")
        print(f"  50%分位数 (中位数): {stats['wishes_stats']['percentiles'][50]:.2f}")
        print(f"  75%分位数: {stats['wishes_stats']['percentiles'][75]:.2f}")
        print(f"  99%分位数: {stats['wishes_stats']['percentiles'][99]:.2f}")

        print(f"\n花费统计 (CNY):")
        print(f"  均值: {stats['spent_stats']['mean']:.2f}")
        print(f"  方差: {stats['spent_stats']['var']:.2f}")
        print(f"  10%分位数: {stats['spent_stats']['percentiles'][10]:.2f}")
        print(f"  25%分位数: {stats['spent_stats']['percentiles'][25]:.2f}")
        print(f"  50%分位数 (中位数): {stats['spent_stats']['percentiles'][50]:.2f}")
        print(f"  75%分位数: {stats['spent_stats']['percentiles'][75]:.2f}")
        print(f"  99%分位数: {stats['spent_stats']['percentiles'][99]:.2f}")

        print(f"\n平均通过星辉兑换的抽数: {stats['avg_glitter_wishes']:.2f}")
        print(f"特殊道具统计: 均值={stats['special_items_stats']['mean']:.4f}")
        print("\n概率分布图已保存为 'genshin_wish_simulation_results.png'")

if __name__ == "__main__":
    main()