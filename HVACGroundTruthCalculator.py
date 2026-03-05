import pandas as pd
import math
import logging
import numpy as np
from typing import Dict, List, Tuple


class HVACGroundTruthCalculator:
    # PA CO2 intensity from EPA eGRID2023 Detailed Data (EPA, 2025)
    EMISSIONS_FACTOR_PA = 0.6458  # lbs CO2/kWh

    # PA residential electricity price from EIA FAQ (EIA, 2022).
    ELECTRICITY_RATE_PA = 0.19  # dollars per/kWh
    SUMMER_COMFORT_RANGE = (73, 79)
    SUMMER_OPTIMAL = 76
    WINTER_COMFORT_RANGE = (68, 75)
    WINTER_OPTIMAL = 70

    # Linear VF for energy cost - equal marginal utility across range
    # Dyer & Sarin (1979): "For monetary attributes with small stakes relative to wealth,
    # linear utility is appropriate" (Management Science 26(8):810-822)
    VF_ENERGY_COST = "linear"

    # Linear VF for environmental impact - physical units have linear marginal value
    # Kotchen & Moore (2007): "When environmental impacts are framed in absolute physical
    # units (tons CO₂, lbs emissions), people exhibit approximately linear preferences"
    # (J. Environmental Economics and Management 54(1):100-123)
    # Log/exponential value function and alpha parameters inspired by Thaler 1999; Heath & Soll 1996; Prelec & Loewenstein 1998.
    VF_ENVIRONMENTAL = "linear"
    VF_COMFORT = "logarithmic, a=1.5"
    VF_PRACTICALITY = "logarithmic, a=1.2"
    def calculate_cooling_load(self, outdoor_temp: float, indoor_temp: float,
                               square_footage: int, r_value: int) -> float:
        """
        Calculate cooling load using ASHRAE cooling load temperature difference method.
        Sources: ASHRAE Standard 55 — Thermal Environmental Conditions for Human Occupancy.
         Wu, W., Skye, H. M., & Domanski, P. A. (2018). Applied Energy, 212, 577-591.

        """
        delta_t = outdoor_temp - indoor_temp

        envelope_area = square_footage * 1.4

        u_factor = 1.0 / r_value

        conductive_load = u_factor * envelope_area * delta_t

        internal_gains = 1000

        window_area = square_footage * 0.15
        solar_gains = window_area * 20

        ventilation_load = conductive_load * 0.20

        total_load = conductive_load + internal_gains + solar_gains + ventilation_load
        print(f"  to Load calculated: {total_load:,.0f} BTU/hr")
        return max(0, total_load)

    def calculate_heating_load(self, outdoor_temp: float, indoor_temp: float,
                               square_footage: int, r_value: int) -> float:
        """
        Calculate heating load using ASHRAE heat loss method.
        Sources: ASHRAE Standard 55 — Thermal Environmental Conditions for Human Occupancy.
         Wu, W., Skye, H. M., & Domanski, P. A. (2018). Applied Energy, 212, 577-591.

        """
        delta_t = indoor_temp - outdoor_temp

        envelope_area = square_footage * 1.4

        u_factor = 1.0 / r_value

        conductive_loss = u_factor * envelope_area * delta_t

        internal_gains = 1000

        infiltration_loss = conductive_loss * 0.25

        total_load = conductive_loss + infiltration_loss - internal_gains
        print(f"  to Load calculated: {total_load:,.0f} BTU/hr")
        return max(0, total_load)

    def calculate_energy_consumption(self, load_btu_hr: float, seer: int,
                                     hvac_age: int, occupancy_context: str, hours: float = 8,
                                     maintenance_level: str = 'moderate') -> float:
        """
        Calculate energy consumption in kWh with age degradation.


        Sources: Domanski, P. A., Henderson, H. I., & Payne, W. V. (2014). NIST TN 1848.
         Fenaughty, K., & Parker, D. (2018). FSEC-PF-474-18, Florida Solar Energy Center.
         Davis Energy Group, Inc. (2010). HVAC Energy Efficiency Maintenance Study (CALMAC).


        Args:
            load_btu_hr: Cooling/heating load (BTU/hr)
            seer: Nameplate SEER rating
            hvac_age: System age (years)
            hours: Operating hours

        Returns:
            Energy consumption in kWh
        """
        # Base degradation rates by maintenance level
        #Rates from Domanski et al. 2014 (NIST TN 1848) and Davis Energy Group 2010.
        maintenance_rates = {
            'good': 0.005,  # 0.5%/year with annual/biannual maintenance
            'moderate': 0.010,  # 1.0%/year with occasional maintenance
            'poor': 0.015  # 1.5%/year with little/no maintenance
        }

        base_rate = maintenance_rates.get(maintenance_level, 0.010)

        # Front-loaded degradation: accelerated first 10 years, slower thereafter
        if hvac_age <= 10:
            # Accelerated early loss (1.5× base rate)
            effective_rate = base_rate * 1.5
            total_degradation = hvac_age * effective_rate
        else:
            # First 10 years at accelerated rate
            early_degradation = 10 * (base_rate * 1.5)
            # Remaining years at slower tail rate (0.5× base rate)
            later_years = hvac_age - 10
            later_degradation = later_years * (base_rate * 0.5)
            total_degradation = early_degradation + later_degradation

        # Cap maximum degradation at 30% (realistic upper bound)
        total_degradation = min(total_degradation, 0.30)

        # Calculate effective SEER after degradation
        effective_seer = seer * (1 - total_degradation)

        print(f"  to SEER degradation: {seer} to {effective_seer:.1f} "
              f"(age={hvac_age}yr, {maintenance_level}, {total_degradation * 100:.1f}% loss)")

        # Convert SEER to EER (approximate relationship)
        eer_estimated = effective_seer * 0.875

        # Calculate power draw
        kw = (load_btu_hr / eer_estimated) / 1000
        if occupancy_context == "occupied_all_day":
            runtime_multiplier = 1.0
        elif occupancy_context.startswith("unoccupied_"):
            hours_away = int(occupancy_context.split("_")[1])
            runtime_multiplier = 1.0 - (hours_away / 24) * 0.5
        elif occupancy_context == "occupied_sleep":
            runtime_multiplier = 0.75
        else:
            runtime_multiplier = 1.0
        total_kwh = kw * hours * runtime_multiplier

        print(f"  to Energy consumption: {total_kwh:.2f} kWh over {hours} hours")
        return total_kwh

    def calculate_comfort_score(self, indoor_temp: float, outdoor_temp: float,
                                household_size: int) -> float:
        """
        Calculate comfort score using ASHRAE 55 with adaptive comfort considerations.

        Citations:
        - Sources:
        ASHRAE Standard 55. Baseline comfort ranges: 73-79F cooling, 68-75F heating.
        Wu, W., et al. (2018). Applied Energy, 212, 577-591. Optimal setpoints.
         de Dear, R., & Brager, G. (2002). Energy and Buildings, 34, 549-561.
        tolerance +/-2F when outdoor 60-85F.
        Wang, Z., & Hong, T. (2020). RSER, 119, 109593.

        """
        if outdoor_temp > 75:
            optimal = 76
            comfort_min, comfort_max = 73, 79
        else:
            optimal = 70
            comfort_min, comfort_max = 68, 75

        if 60 < outdoor_temp < 85:
            comfort_min -= 2
            comfort_max += 2

        deviation = abs(indoor_temp - optimal)

        if comfort_min <= indoor_temp <= comfort_max:
            comfort_score = 10 - (deviation)
        else:
            if indoor_temp < comfort_min:
                range_violation = comfort_min - indoor_temp
            else:
                range_violation = indoor_temp - comfort_max
            # Partial comfort score. Wang & Hong 2020, RSER, 119, 109593.
            comfort_score = 6 - (range_violation)

        if household_size > 3:
            size_penalty = (household_size - 3) * 0.3
            comfort_score -= size_penalty * (deviation / 3)

        return max(0, min(10, comfort_score))

    def calculate_practicality_score(self, outdoor_temp: float, indoor_temp: float,) -> float:
        """
        Calculate practicality as likelihood of sustained behavioral adoption.
        NOT about comfort (that's the comfort criterion), but about behavioral abandonment.

        Citations:
        - Xu et al. (2017). "Investigating willingness to save energy and communication about
          energy use in the American workplace with the attitude-behavior-context model"
          Energy Research & Social Science 32:13-22
          Finding: Override behavior increases with extreme setpoints regardless of comfort

        - Karjalainen (2007). "Gender differences in thermal comfort and use of thermostats"
          Indoor Air 17(1):60-67
          Finding: Habituation difficulty for non-standard temperatures drives abandonment
        """

        if outdoor_temp > 75:  # Cooling mode
            if indoor_temp >= 82:
                # INCREASED: 1.0 to 1.5 per degree above 82°F
                extremity_penalty = (indoor_temp - 82) * 1.5
            elif indoor_temp <= 71:
                # INCREASED: 0.6 to 1.0 per degree below 71°F
                extremity_penalty = (71 - indoor_temp) * 1.0
            else:
                extremity_penalty = 0
        else:  # Heating mode
            if indoor_temp <= 63:
                # INCREASED: 1.0 to 1.8 per degree below 63°F
                extremity_penalty = (63 - indoor_temp) * 1.8
            elif indoor_temp >= 76:
                # INCREASED: 0.5 to 0.8 per degree above 76°F
                extremity_penalty = (indoor_temp - 76) * 0.8
            else:
                extremity_penalty = 0

        base_score = 10 - extremity_penalty

        # LOWER FLOOR: 1.5 to  0.5 to allow more penalty
        base_score = max(0.5, base_score)

        # Component 2: change in T operational feasibility
        # Large ΔT indicates system operating at limits; lower reliability/higher failure risk
        delta_t = abs(outdoor_temp - indoor_temp)
        if delta_t < 10:
            delta_t_multiplier = 1.0
        elif delta_t < 20:
            delta_t_multiplier = 0.95
        elif delta_t < 35:
            delta_t_multiplier = 0.85
        else:
            delta_t_multiplier = 0.70

        base_score *= delta_t_multiplier

        return max(0.0, min(10.0, base_score))

    def calculate_monthly_cost(self, per_period_cost: float, periods_per_month: int = 90) -> float:
        """
        Convert per-8hr-period cost to estimated monthly cost.

        Args:
            per_period_cost: Cost per 8-hour period ($)
            periods_per_month: How many 8-hour periods per month (default 30 days)

        Returns:
            Estimated monthly cost in dollars
        """
        return per_period_cost * periods_per_month

    def calculate_budget_penalty(self, monthly_cost: float, monthly_budget: float) -> float:
        """
        Calculate budget constraint penalty multiplier for energy cost score.

        - <80%  : No penalty. Thaler, R. (1999). J. Behavioral Decision Making, 12, 183-206.
    - 80-100%: Linear decline. Heath, C., & Soll, J. B. (1996). J. Consumer Research, 23(1), 40-52.
    - 100-150%: Exponential decline. Prelec & Loewenstein (1998). Marketing Science, 17(1), 4-28.
            Energy-specific: Heutel, G. (2017). NBER WP 23692.
    - >150%  : Eliminated. Gathergood, J. (2012). J. Economic Psychology, 33(3), 590-602.


        Args:
            monthly_cost: Estimated monthly energy cost for this alternative
            monthly_budget: User's monthly utility budget

        Returns:
            Penalty multiplier (0.0 to 1.0) to apply to energy cost score
        """
        if monthly_budget <= 0:
            return 1.0  # No budget constraint

        utilization = monthly_cost / monthly_budget

        if utilization < 0.80:
            # if utilization < 0.80:    # Safety margin. Thaler 1999.
            return 1.0

        elif utilization < 1.0:
            # Linear decline. Heath & Soll 1996.
            return 1.0 - 2.5 * (utilization - 0.80)

        elif utilization < 1.5:
            # Exponential loss aversion. Prelec & Loewenstein 1998; Heutel 2017.
            import math
            return 0.5 * math.exp(-3.0 * (utilization - 1.0))

        else:
            #Eliminated. Gathergood 2012.
            return 0.0

    def apply_value_function(self, raw_value: float, vf_spec: str, value_type: str) -> float:
        """
        Apply value function transformation to raw criterion values.

        Reference ranges derived from:
        - Huyen & Cetin (2019): Baseline consumption
        - Kim et al. (2024): Setpoint sensitivity
        - Cetin & Novoselac (2015): Runtime patterns
        - Alves et al. (2016): Degradation multipliers
        - Krarti & Howarth (2020): SEER-power relationships
        - EPA eGRID (2025): Grid emissions factors
        """
        reference_ranges = {
                'energy_cost': {
        # 5th-95th percentile from actual dataset distribution
        # Captures 90% of realistic alternatives, creates sensitivity in cluster region

                    # Min (efficient): Huyen & Cetin (2019), Energies 12(1):188;
                    #   Kim et al. (2024), Building Simulation; Cetin & Novoselac (2015), EB 96:210.

                    'min': 0.47,


        # Max (degraded): Alves et al. (2016), EB 130:408; Krarti & Howarth (2020), JBE 31:101457.

        'max': 3.31,
        'decreasing': True
    },
    'environmental': {
        # Derived from energy bounds x PA emissions factor.
        # # Source: EPA eGRID2023 Detailed Data (EPA, 2025).
        # Formula: (cost / electricity_rate) × emissions_factor
        # Min: (0.47 / 0.19) × 0.6458 = 2.474 × 0.6458 = 1.60 lbs CO₂
        # Max: (3.31 / 0.19) × 0.6458 = 17.421 × 0.6458 = 11.25 lbs CO₂
        'min': 1.60,
        'max': 11.25,
        'decreasing': True
    },
            'comfort': {
                'min': 0.0,
                'max': 10.0,
                'decreasing': False
            },
            'practicality': {
                'min': 1.5,
                'max': 10.0,
                'decreasing': False
            }
        }

        ref = reference_ranges[value_type]
        x_min = ref['min']
        x_max = ref['max']

        x = raw_value

        vf_type = vf_spec.split(',')[0].strip().lower()

        # Normalize (now can go outside [0,1] range)
        if ref['decreasing']:
            x_normalized = (x_max - x) / (x_max - x_min)
        else:
            x_normalized = (x - x_min) / (x_max - x_min)

        # Apply transformation
        if vf_type == 'linear':
            u_x = x_normalized

        elif vf_type == 'polynomial':
            try:
                a = float([p for p in vf_spec.split(',') if 'a=' in p][0].split('=')[1].strip())
            except:
                a = 1.0
            u_x = x_normalized ** a

        elif vf_type == 'exponential':
            try:
                a = float([p for p in vf_spec.split(',') if 'a=' in p][0].split('=')[1].strip())
            except:
                a = 1.0
            if a == 0:
                u_x = x_normalized
            else:
                u_x = (1 - math.exp(a * x_normalized)) / (1 - math.exp(a))

        elif vf_type == 'logarithmic':
            try:
                a = float([p for p in vf_spec.split(',') if 'a=' in p][0].split('=')[1].strip())
            except:
                a = 1.0
            if a == -1:
                u_x = x_normalized
            else:
                # Handle negative x_normalized (better than best case)
                if a * x_normalized + 1 <= 0:
                    u_x = 0.0
                else:
                    u_x = math.log(a * x_normalized + 1) / math.log(a + 1)

        else:
            u_x = x_normalized

        # NOW clamp the final score to [0, 10]
        return max(0.0, min(10.0, u_x * 10.0))

    def calculate_scenario_scores(self, scenario: Dict) -> Dict:
        """
        Calculate complete ground truth scores for a scenario with all alternatives.
        Feeds raw criterion values directly to value functions per MAVT principles.
        """


        is_cooling = scenario['outdoor_temp'] > 75

        raw_results = {}

        for alt in scenario['alternatives']:
            if isinstance(alt, str):
                import re

                # Enhanced parsing for "Off" alternatives
                # Handles: "Off", "Off (55)", "Off (let drift to 85)", etc.
                if 'off' in alt.lower():
                    # Priority 1: Number in parentheses "Off (85)"
                    paren_match = re.search(r'\(.*?(\d+).*?\)', alt)
                    if paren_match:
                        effective_temp = float(paren_match.group(1))
                    # Priority 2: Number after "to" keyword "drift to 85"
                    elif 'to' in alt.lower():
                        to_match = re.search(r'to\s+(\d+)', alt, re.IGNORECASE)
                        if to_match:
                            effective_temp = float(to_match.group(1))
                        else:
                            # Fallback to drift calculation
                            if is_cooling:
                                effective_temp = scenario['outdoor_temp'] - 5
                            else:
                                effective_temp = scenario['outdoor_temp'] + 5
                    # Priority 3: No number specified - use drift
                    else:
                        if is_cooling:
                            effective_temp = scenario['outdoor_temp'] - 5
                        else:
                            effective_temp = scenario['outdoor_temp'] + 5
                else:
                    # Not an "off" alternative - extract first number found
                    numbers = re.findall(r'\d+', alt)
                    if numbers:
                        effective_temp = float(numbers[0])
                    else:
                        print(f"   Could not parse alternative: {alt}")
                        continue
            else:
                effective_temp = float(alt)

            if is_cooling:
                load = self.calculate_cooling_load(
                    scenario['outdoor_temp'],
                    effective_temp,
                    scenario['square_footage'],
                    scenario['r_value']
                )
            else:
                load = self.calculate_heating_load(
                    scenario['outdoor_temp'],
                    effective_temp,
                    scenario['square_footage'],
                    scenario['r_value']
                )

            kwh = self.calculate_energy_consumption(
                load,
                scenario['seer'],
                scenario['hvac_age'],
                occupancy_context=scenario.get('occupancy_context', 'occupied_all_day'),
                maintenance_level=scenario.get('maintenance_level', 'moderate')
            )

            energy_cost = kwh * scenario.get('electricity_rate', self.ELECTRICITY_RATE_PA)
            emissions = kwh * self.EMISSIONS_FACTOR_PA


            comfort = self.calculate_comfort_score(
                effective_temp,
                scenario['outdoor_temp'],
                scenario['household_size']
            )

            practicality = self.calculate_practicality_score(
                scenario['outdoor_temp'],
                effective_temp,
            )
            raw_results[alt] = {
                'kwh': kwh,
                'energy_cost_dollars': energy_cost,
                'emissions_lbs': emissions,
                'comfort_raw': comfort,
                'practicality_raw': practicality
            }

        final_scores = {}

        for alt, raw in raw_results.items():


            try:
                energy_vf = self.apply_value_function(
                    raw['energy_cost_dollars'],
                    self.VF_ENERGY_COST,
                    'energy_cost'
                )

            except Exception as e:
                print(f"  ✗ Energy VF ERROR: {e}")
                energy_vf = 5.0

            try:
                env_vf = self.apply_value_function(
                    raw['emissions_lbs'],
                    self.VF_ENVIRONMENTAL,
                    'environmental'
                )

            except Exception as e:
                print(f"  ✗ Environmental VF ERROR: {e}")
                env_vf = 5.0

            try:
                comfort_vf = self.apply_value_function(
                    raw['comfort_raw'],
                    self.VF_COMFORT,
                    'comfort'
                )

            except Exception as e:
                print(f"  ✗ Comfort VF ERROR: {e}")
                comfort_vf = raw['comfort_raw']

            try:
                practicality_vf = self.apply_value_function(
                    raw['practicality_raw'],
                    self.VF_PRACTICALITY,
                    'practicality'
                )

            except Exception as e:
                print(f"  ✗ Practicality VF ERROR: {e}")
                practicality_vf = raw['practicality_raw']

            # Apply budget penalty if budget constraint exists
            if 'utility_budget' in scenario and scenario['utility_budget'] > 0:
                # Convert 8-hour cost to monthly estimate (30 days)
                monthly_cost = self.calculate_monthly_cost(
                    raw['energy_cost_dollars'],
                    periods_per_month=90 # 24 hours per day divided by 8 hour decision period
                )

                budget_penalty = self.calculate_budget_penalty(
                    monthly_cost,
                    scenario['utility_budget']
                )

                # Apply penalty to energy cost score
                energy_vf_penalized = energy_vf * budget_penalty

                print(f"  Budget check: ${monthly_cost:.2f}/month vs ${scenario['utility_budget']:.2f} budget")
                print(
                    f"  Utilization: {monthly_cost / scenario['utility_budget'] * 100:.1f}% to penalty: {budget_penalty:.3f}")
                print(f"  Energy score: {energy_vf:.2f} to {energy_vf_penalized:.2f} (after penalty)")

                energy_vf = energy_vf_penalized

            final_scores[alt] = {
                'energy_cost_score': round(energy_vf, 2),
                'environmental_score': round(env_vf, 2),
                'comfort_score': round(comfort_vf, 2),
                'practicality_score': round(practicality_vf, 2),
                'raw_kwh': round(raw['kwh'], 2),
                'raw_cost': round(raw['energy_cost_dollars'], 2),
                'raw_emissions': round(raw['emissions_lbs'], 2)
            }

            print(f"  to FINAL SCORES:")
            print(
                f"     Energy: {energy_vf:.2f}, Environmental: {env_vf:.2f}, Comfort: {comfort_vf:.2f}, Practicality: {practicality_vf:.2f}\n")

        return final_scores


def process_hvac_scenarios(csv_filename: str = "HVACScenarios.csv",  output_filename: str = "ground_truth_hvac.csv"):
    """
    Read HVAC scenarios from CSV and calculate ground truth scores for all alternatives.

    Args:
        csv_filename: Path to CSV file with scenarios
        output_filename: Where to save ground truth results

    Expected CSV columns:
        Question, Location, Square Footage, Insulation, Household Size,
        Utility Budget, Housing Type, Outdoor Temp, House Age, R-Value,
        HVAC Age, SEER, Alternative 1, Alternative 2, Alternative 3
    """

    df = pd.read_csv(csv_filename)

    print(f"Found {len(df)} scenarios")

    calculator = HVACGroundTruthCalculator()

    results = []

    for idx, row in df.iterrows():
        print(f"Processing scenario {idx + 1}/{len(df)}: {row['Location']}")
        electricity_rate = 0.19

        alternatives = []
        for alt_col in ['Alternative 1', 'Alternative 2', 'Alternative 3']:
            alt_val = str(row[alt_col]).strip()

            if pd.isna(row[alt_col]) or alt_val == '' or alt_val == 'nan':
                continue
            alternatives.append(alt_val)

        scenario = {
            'question': row['Question'],
            'location': row['Location'],
            'square_footage': int(row['Square Footage']),
            'r_value': int(row['R-Value']),
            'household_size': int(row['Household Size']),
            'outdoor_temp': float(row['Outdoor Temp']),
            'seer': int(row['SEER']),
            'hvac_age': int(row['HVAC Age']),
            'occupancy_context': row['Occupancy Context'] if 'Occupancy Context' in row.index else 'occupied_all_day',
            'electricity_rate': electricity_rate,
            'alternatives': alternatives,
        }
        try:
            scores = calculator.calculate_scenario_scores(scenario)
            alts_for_ranking = [
                {
                    "alternative": alt,
                    "energy_cost": scores[alt]["energy_cost_score"],
                    "environmental": scores[alt]["environmental_score"],
                    "comfort": scores[alt]["comfort_score"],
                    "practicality": scores[alt]["practicality_score"]
                }
                for alt in scores
            ]
            ranking_result = apply_mavt_ranking(alts_for_ranking)
            for alt, alt_scores in scores.items():
                result_row = {
                    'scenario_id': idx,
                    'question': row['Question'],
                    'location': row['Location'],
                    'square_footage': row['Square Footage'],
                    'insulation': row.get('Insulation', ''),
                    'household_size': row['Household Size'],
                    'utility_budget': row.get('Utility Budget', ''),
                    'housing_type': row.get('Housing Type', ''),
                    'outdoor_temp': row['Outdoor Temp'],
                    'house_age': row.get('House Age', ''),
                    'alternative': alt,
                    'energy_cost_score': alt_scores['energy_cost_score'],
                    'environmental_score': alt_scores['environmental_score'],
                    'comfort_score': alt_scores['comfort_score'],
                    'practicality_score': alt_scores['practicality_score'],
                    'mavt_score': ranking_result["weighted_scores"][list(scores.keys()).index(alt)],
                    'rank': ranking_result["ranks"][list(scores.keys()).index(alt)],
                    'raw_kwh': alt_scores['raw_kwh'],
                    'raw_cost': alt_scores['raw_cost'],
                    'raw_emissions': alt_scores['raw_emissions']
                }
                results.append(result_row)

        except Exception as e:
            print(f"ERROR processing scenario {idx}: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)

    print(f"\nGround truth saved to {output_filename}")
    print(f"Total alternatives scored: {len(results_df)}")
    return results_df

def apply_mavt_ranking(alternatives_scores: List[Dict]) -> Dict:
    """
    Apply MAVT weighted sum to rank alternatives

    Args:
        alternatives_scores: List of dicts with keys: alternative, energy_cost, environmental, comfort, practicality

    Returns:
        Dict with ranked_alternatives, ranks, weighted_scores
    """
    try:
        alternatives = [alt["alternative"] for alt in alternatives_scores]

        # Calculate weighted sum for each alternative
        weighted_scores = []
        for alt_scores in alternatives_scores:
            weighted_sum = (
                    CRITERION_WEIGHTS["energy_cost"] * alt_scores["energy_cost"] +
                    CRITERION_WEIGHTS["environmental"] * alt_scores["environmental"] +
                    CRITERION_WEIGHTS["comfort"] * alt_scores["comfort"] +
                    CRITERION_WEIGHTS["practicality"] * alt_scores["practicality"]
            )
            weighted_scores.append(weighted_sum)

        # Rank alternatives (higher weighted sum = better = lower rank number)
        ranked_indices = np.argsort(weighted_scores)[::-1]  # Descending order
        ranked_alternatives = [alternatives[i] for i in ranked_indices]

        # Create rank numbers (1 = best, 2 = second, 3 = third)
        ranks = [0] * len(alternatives)
        for rank_position, alt_index in enumerate(ranked_indices):
            ranks[alt_index] = rank_position + 1

        return {
            "ranked_alternatives": ranked_alternatives,
            "ranks": ranks,
            "weighted_scores": weighted_scores
        }

    except Exception as e:
        logging.error(f"MAVT ranking failed: {e}")

        # Fallback: rank by average score
        avg_scores = []
        for alt_scores in alternatives_scores:
            avg = np.mean([
                alt_scores["energy_cost"],
                alt_scores["environmental"],
                alt_scores["comfort"],
                alt_scores["practicality"]
            ])
            avg_scores.append(avg)

        ranked_indices = np.argsort(avg_scores)[::-1]
        ranked_alternatives = [alternatives[i] for i in ranked_indices]

        ranks = [0] * len(alternatives)
        for rank_position, alt_index in enumerate(ranked_indices):
            ranks[alt_index] = rank_position + 1

        return {
            "ranked_alternatives": ranked_alternatives,
            "ranks": ranks,
            "weighted_scores": avg_scores,
            "error": str(e)
        }


CRITERION_WEIGHTS = {
    "energy_cost": 0.30,
    "environmental": 0.35,
    "comfort": 0.20,
    "practicality": 0.15
}
if __name__ == "__main__":
    process_hvac_scenarios(
        csv_filename="HVACScenarios.csv",
        output_filename="ground_truth_hvac.csv"
    )
