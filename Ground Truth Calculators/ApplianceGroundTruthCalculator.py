import pandas as pd
import math
import logging
import numpy as np
from typing import Dict, List, Tuple
class ApplianceGroundTruthCalculator:
    # PA CO2 intensity from EPA eGRID2023 Detailed Data (EPA, 2025)
    EMISSIONS_FACTOR_PA = 0.6458  # lbs CO2/kWh

    # PA residential electricity price from EIA FAQ (EIA, 2022).
    ELECTRICITY_RATE_PA = 0.19  # dollars per/kWh
    # Peak window 2 PM-6 PM from PECO Energy TOU documentation (PECO, 2021).
    PEAK_HOURS = (14, 18)

    APPLIANCE_NOISE_LEVELS = {
        'dishwasher': 45,
        'washer': 50,
        'dryer': 55
    }


    NOISE_LIMIT_DAYTIME = 45     # dBA acceptable during day
    NOISE_LIMIT_EVENING = 35     # dBA acceptable after 10pm
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
    def determine_rate_period(self, run_time_hour: int) -> str:
        """
        Determine if run time falls in peak or off-peak period.

        Args:
            run_time_hour: Hour of day (0-23, e.g., 19 for 7pm)

        Returns:
            "peak" or "offpeak"

        """
        peak_start, peak_end = self.PEAK_HOURS

        if peak_start <= run_time_hour < peak_end:
            return "peak"

        return "offpeak"

    def calculate_energy_cost(self, kwh_cycle: float, run_time_hour: int,
                             peak_rate: float, offpeak_rate: float) -> float:
        """
        Calculate energy cost based on TOU rate structure.

        args:
            kwh_cycle: Fixed energy per cycle
            run_time_hour: When appliance runs (0-23)
            peak_rate: Peak period $/kWh
            offpeak_rate: Off-peak period $/kWh

        Returns:
            Energy cost in dollars
       Dryer (heat pump 0.8-1.5 kWh; standard electric 2.5-4.0 kWh):
       Winfield, D., et al. (2016). Measured Performance of Heat Pump Clothes Dryers.
        ACEEE Summer Study on Energy Efficiency in Buildings.
        Northeast Energy Efficiency Partnerships (NEEP). (2015).
        New Study Unearths Energy Baseline for Clothes Dryers in the Northeast.
        Dishwasher (0.9-1.1 kWh Energy Star): Porras, G., et al. (2020). ERC, 2.
        Washer (0.15-0.25 kWh HE): Chen-Yu, J., & Emmel, J. (2018). Fashion & Textiles, 5.
        Hustvedt, G., Ahn, M., & Emmel, J. (2013). Int. J. Consumer Studies, 37.

        """
        period = self.determine_rate_period(run_time_hour)

        if period == "peak":
            rate = peak_rate
        else:
            rate = offpeak_rate

        cost = kwh_cycle * rate
        print(f" Energy cost: {kwh_cycle} kWh × ${rate:.4f}/kWh ({period}) = ${cost:.4f}")
        return cost

    def calculate_environmental_impact(self, kwh_cycle: float) -> float:
        """
        Calculate CO2 emissions from electricity consumption.

       
        Args:
            kwh_cycle: Energy consumption per cycle

        Returns:
            CO2 emissions in pounds

        Citation: EPA eGRID (2023), Pennsylvania grid emissions factor
        0.6458 lbs CO2/kWh (state-level average)
        """
        emissions = kwh_cycle * self.EMISSIONS_FACTOR_PA
        print(f"  : Emissions: {kwh_cycle} kWh × {self.EMISSIONS_FACTOR_PA} lbs/kWh = {emissions:.3f} lbs CO2")
        return emissions

    def calculate_comfort_score(self, delay_hours: float, run_time_hour: int,
                               housing_type: str, occupants: int,
                               appliance_type: str) -> float:
        """
        Calculate comfort score based on delay inconvenience and noise disruption.

        Components:
        1. Delay penalty (longer delay = more inconvenience)
        2. Noise disruption (late night in apartment = worse)
        3. Household size multiplier (more people = dishes/laundry pile up faster)

        Source: Paetz, A., Dutschke, E., & Fichtner, W. (2012). Shifting dish-washer use.
        ACEEE Summer Study on Energy Efficiency in Buildings, Paper 0193-000232.
        12-hour delay is the max acceptable without lifestyle disruption.
 """

        # Component 1: Base delay penalty
        # Paetz et al.: 12hr delay is maximum acceptable for dishwasher
        if delay_hours == 0:
            base_comfort = 10.0
        elif delay_hours <= 3:
            base_comfort = 8.0   # Short delay, minor inconvenience
        elif delay_hours <= 7:
            base_comfort = 6.0   # Medium delay, moderate inconvenience
        elif delay_hours <= 12:
            base_comfort = 4.0   # Long delay but still within "acceptable" range
        else:
            base_comfort = 2.0   # Beyond acceptable (>12hr)

        print(f"  : Base comfort (delay={delay_hours}hr): {base_comfort}/10")

        # Component 2: Noise disruption penalty
        # Depends on: time of day + housing type + appliance noise

        if appliance_type.lower() == "dishwasher":
            appliance_noise = 45
        elif appliance_type.lower() == "washer" or "washing" in appliance_type.lower():
            appliance_noise = 50
        elif appliance_type.lower() == "dryer":
            appliance_noise = 55
        else:
            appliance_noise = 50
        noise_penalty = 0.0
            # Late night running (10pm-7am)
        if 22 <= run_time_hour or run_time_hour < 7:
           if appliance_noise > self.NOISE_LIMIT_EVENING:
                noise_penalty = 2.0  # Base penalty for late night

                # Housing type multiplier
                # Apartment noise complaints from shared walls
                if housing_type == "Apartment":
                    noise_penalty *= 1.5   # Neighbors very close
                elif housing_type == "Townhouse" or housing_type == "Rowhouse":
                    noise_penalty *= 1.2   # Shared walls
                else:  # Single-family
                    noise_penalty *= 0.8   # Isolated, lower concern

                print(f"  : Noise penalty (late night, {housing_type}): -{noise_penalty:.1f}")

        # Component 3: Household size impact
        # Larger households : dishes pile up faster : delay worse
        if occupants >= 5:
            size_penalty = 1.5
        elif occupants >= 3:
            size_penalty = 0.8
        else:
            size_penalty = 0.0

        # Apply size penalty proportional to delay
        # (No delay = no penalty, long delay = full penalty)
        size_penalty *= (delay_hours / 12.0)  # Scale by delay fraction
        print(f"  : Household size penalty ({occupants} occupants): -{size_penalty:.1f}")

        final_comfort = base_comfort - noise_penalty - size_penalty
        return max(0.0, min(10.0, final_comfort))

    def calculate_practicality_score(self, delay_hours: float, run_time_hour: int,
                                    housing_type: str, occupants: int,
                                    appliance_type: str) -> float:
        """
        Calculate practicality as behavioral adoption likelihood.

        NOT about comfort (that's comfort criterion), but about:
        - Willingness to adopt TOU scheduling behavior
        - Complexity of remembering to delay
        - Household coordination difficulty

        Citations:
        - Paetz, A., Dutschke, E., & Fichtner, W. (2012). ACEEE, Paper 0193-000232.
- Indonesia TOU Adoption Study. (2024). PMC11190461.
- Shewale, A., et al. (2023). Arabian Journal for Science and Engineering.
- Newsham, G. R., & Bowker, B. G. (2010). Energy Policy, 38(7), 3289-3296.
  General TOU/CPP load-shifting context.
- Waseem, M., et al. (2020). Electric Power Systems Research, 187, 106477.
  Optimization-based appliance scheduling context.

  """

        # Component 1: Base adoption likelihood by delay duration
        # Paetz: 12hr delay acceptable, but adoption varies
        # Shewale: <20% adoption for manual TOU scheduling

        if delay_hours == 0:
            base_practicality = 10.0  # No behavior change required
        elif delay_hours <= 2:
            base_practicality = 8.0   # Minor behavior change, high adoption
        elif delay_hours <= 4:
            base_practicality = 6.5   # Moderate change, medium adoption
        elif delay_hours <= 8:
            base_practicality = 4.5   # Significant delay, lower adoption
        elif delay_hours <= 12:
            base_practicality = 3.0   # Maximum acceptable (Paetz), but low adoption
        else:
            base_practicality = 1.5   # Beyond typical adoption range

        print(f"  : Base practicality (delay={delay_hours}hr): {base_practicality}/10")

        # Component 2: Timing complexity (remembering to run at specific time)
        # Late night/early morning = harder to remember/coordinate
        timing_penalty = 0.0

        # Paetz et al.: "If low-price zones applied on brink of day, it was
        # perceived as too early or too late"
        if 0 <= run_time_hour < 6:  # Middle of night (midnight-6am)
            timing_penalty = 2.0   # Very inconvenient timing
        elif 22 <= run_time_hour < 24:  # Late night (10pm-midnight)
            timing_penalty = 1.0   # Somewhat inconvenient

        print(f"  : Timing complexity penalty: -{timing_penalty:.1f}")

        # Component 3: Household coordination difficulty
        # More occupants = harder to coordinate "don't run dishes yet"
        # Ground Truth Data Section 9: Behavioral barriers increase with complexity

        if occupants >= 5:
            coordination_penalty = 1.5
        elif occupants >= 3:
            coordination_penalty = 0.8
        else:
            coordination_penalty = 0.0

        # Scale penalty by delay (longer delay = more coordination needed)
        coordination_penalty *= (delay_hours / 12.0)
        print(f"  : Coordination penalty ({occupants} occupants): -{coordination_penalty:.1f}")

        final_practicality = base_practicality - timing_penalty - coordination_penalty
        DAYTIME_START = 7  # 7am
        DAYTIME_END = 22  # 10pm

        if DAYTIME_START <= run_time_hour < DAYTIME_END:
            final_practicality = max(final_practicality, 4)

        return max(1.5, min(10.0, final_practicality))


    def parse_alternative(self, alt: str, scenario: Dict) -> Tuple[int, float]:
        """
        Parse alternative text to extract run time and delay.

        Now uses scenario-provided baseline time instead of hardcoded defaults.
        This allows user flexibility and makes the baseline visible to AI.

        Args:
            alt: Alternative text string (e.g., "Run at 7pm")
            scenario: Full scenario dict (must contain 'Baseline Time' key)

        Returns:
            (run_time_hour, delay_hours)
        """
        import re

        # Extract run time from alternative (e.g., "7pm", "10pm", "2am")
        time_match = re.search(r'(\d{1,2})(?::\d{2})?\s*(am|pm)', alt, re.IGNORECASE)
        if not time_match:
            print(f"  : Could not parse run time from: {alt}")
            # Return baseline with no delay
            baseline_hour = self._parse_time_to_hour(scenario.get('Baseline Time', '7pm'))
            return baseline_hour, 0.0

        hour = int(time_match.group(1))
        am_pm = time_match.group(2).lower()

        # Convert to 24-hour format
        if am_pm == "pm" and hour != 12:
            run_time_hour = hour + 12
        elif am_pm == "am" and hour == 12:
            run_time_hour = 0
        else:
            run_time_hour = hour

        # Parse baseline time from scenario
        baseline_str = scenario.get('Baseline Time', '7pm')
        baseline_hour = self._parse_time_to_hour(baseline_str)

        # Calculate delay from baseline
        if run_time_hour >= baseline_hour:
            delay_hours = float(run_time_hour - baseline_hour)
        else:
            # Crossed midnight
            delay_hours = float(24 - baseline_hour + run_time_hour)

        print(f"  Parsed: '{alt}' : run at {run_time_hour:02d}:00, "
              f"delay={delay_hours}hr from baseline {baseline_str}")

        return run_time_hour, delay_hours
    def _parse_time_to_hour(self, time_str: str) -> int:
        """
        Helper function to convert time string to 24-hour format.

        Examples:
        - "7pm" : 19
        - "8am" : 8
        - "12pm" : 12
        - "12am" : 0

        Args:
            time_str: Time string (e.g., "7pm", "8am")

        Returns:
            Hour in 24-hour format (0-23)
        """
        import re

        match =re.search(r'(\d{1,2})(?::\d{2})?\s*(am|pm)', time_str, re.IGNORECASE)
        if not match:
            # Default to 7pm if unparseable
            print(f"  : Could not parse baseline time '{time_str}', defaulting to 7pm")
            return 19

        hour = int(match.group(1))
        am_pm = match.group(2).lower()

        if am_pm == "pm" and hour != 12:
            return hour + 12
        elif am_pm == "am" and hour == 12:
            return 0
        else:
            return hour

    def apply_value_function(self, raw_value: float, vf_spec: str, value_type: str) -> float:
        """
        Apply value function transformation to raw criterion values.

        EXACT SAME METHOD AS HVAC - maintains consistency across calculators.

        Reference ranges derived from actual scenario distribution (5th-95th percentile).

        Args:
            raw_value: Raw criterion value (e.g., dollars, lbs CO2)
            vf_spec: Value function specification (e.g., "linear", "logarithmic, a=1.5")
            value_type: Criterion name for reference range lookup

        Returns:
            Transformed score on 0-10 scale
        """
        reference_ranges = {
            'energy_cost': {
                # Reference range derived from representative appliance usage:
                # Min: Efficient HE washer off-peak≈0.1 kWh × $0.09/kWh ≈ $0.01 (rounded to $0.02 for 5th percentile)
                # Max: Standard electric resistance dryer at peak≈4.5 kWh × $0.20/kWh ≈ $0.90
                # Sources: Winfield et al. (2016); NEEP (2015); Porras et al. (2020); Chen-Yu & Emmel (2018); EIA (2022) for PA electricity price.

                'min': 0.02,
                'max': 0.90,
                'decreasing': True
            },
            'environmental': {
                # Derived from energy bounds × PA emissions factor:
                # Min: 0.1 kWh × 0.6458 lbs/kWh ≈ 0.065 lbs CO2 (adjusted to data set; none of my alternatives aligned with the 5th percintile or lower, so adjusted up)
                # Max: 4.5 kWh × 0.6458 lbs/kWh ≈ 2.9 lbs CO2 (adjusted to data set; more extreme alternatives included (>95th percentile) so bound adjusted up)
                # Source: EPA eGRID2023 Detailed Data (Version 2).

                'min': 0.09,
                'max': 3.83,
                'decreasing': True
            },
            'comfort': {
                'min': 0.0,
                'max': 10.0,
                'decreasing': False
            },
            'practicality': {
                'min': 1.5,  # Floor from calculation
                'max': 10.0,
                'decreasing': False
            }
        }

        ref = reference_ranges[value_type]
        x_min = ref['min']
        x_max = ref['max']

        # Use raw_value directly - allow extrapolation (following HVAC pattern)
        x = raw_value

        vf_type = vf_spec.split(',')[0].strip().lower()

        # Normalize (can go outside [0,1] range)
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
                    u_x = 0.0  # absolutely horrible score
                else:
                    u_x = math.log(a * x_normalized + 1) / math.log(a + 1)

        else:
            u_x = x_normalized

        # Clamp final score to [0, 10]
        return max(0.0, min(10.0, u_x * 10.0))

    def calculate_budget_penalty(self, monthly_cost: float, monthly_budget: float) -> float:
        """
        - <80%  : No penalty. Thaler, R. (1999). J. Behavioral Decision Making, 12, 183-206.
    - 80-100%: Linear decline. Heath, C., & Soll, J. B. (1996). J. Consumer Research, 23(1), 40-52.
    - 100-150%: Exponential decline. Prelec & Loewenstein (1998). Marketing Science, 17(1), 4-28.
            Energy-specific: Heutel, G. (2017). NBER WP 23692.
    - >150%  : Eliminated. Gathergood, J. (2012). J. Economic Psychology, 33(3), 590-602.

        Args:
            monthly_cost: Estimated monthly energy cost for this alternative ($)
            monthly_budget: User's stated monthly utility budget ($)

        Returns:
            Penalty multiplier ∈ [0.0, 1.0] to apply to energy cost score

        Examples:
            Budget = $175/month
            - $140 cost (80%):  penalty = 1.0   (no reduction)
            - $158 cost (90%):  penalty = 0.75  (mild reduction)
            - $175 cost (100%): penalty = 0.5   (moderate reduction)
            - $193 cost (110%): penalty = 0.37  (significant reduction)
            - $219 cost (125%): penalty = 0.14  (severe reduction)
            - $263 cost (150%): penalty = 0.01  (near elimination)
            - $267 cost (153%): penalty = 0.0   (complete elimination)
        """

        utilization = monthly_cost / monthly_budget

        if utilization < 0.80:
            # Thaler (1999): Mental budget safety margin
            return 1.0

        elif utilization < 1.0:
            # 80-100%: Linear decline (approaching budget limit)
            #Linear decline. Heath & Soll 1996.
            return 1.0 - 2.5 * (utilization - 0.80)

        elif utilization < 1.5:
            # 100-150%: Exponential decline (budget violation)
            # Exponential loss aversion. Prelec & Loewenstein 1998; Heutel 2017.
            import math
            return 0.5 * math.exp(-3.0 * (utilization - 1.0))

        else:
            # >150%: Complete elimination (infeasibility threshold)
            # Gathergood (2012)
            return 0.0

    def calculate_monthly_cost(self, per_cycle_cost: float, cycles_per_month: int = 30) -> float:
        """
        Convert per-cycle cost to monthly cost.
        Standard: 30 cycles/month for dishwasher/washer/dryer.
        Citations: Porras et al. (2020), Chen-Yu & Emmel (2018)
        """
        return per_cycle_cost * cycles_per_month

    def calculate_scenario_scores(self, scenario: Dict) -> Dict:
        """
        Calculate complete ground truth scores for appliance scenario with all alternatives.

        Expected scenario structure (EXACT MATCH to CSV parameters):
        {
            'Description': "When should I run my dishwasher after dinner tonight?",
            'Location': "Philadelphia, PA",
            'Utility Budget': 150,
            'Appliance': "dishwasher",
            'Housing Type': "Apartment",
            'Occupants': 2,
            'Peak Rate': 0.18,      # Dollar amount, not string
            'Off-Peak Rate': 0.09,  # Dollar amount, not string
            'kwh/cycle': 1.25,
            'Appliance Age/Type': "7 years",
            'Alternative 1': "Run at 7pm",
            'Alternative 2': "Run at 10pm",
            'Alternative 3': "Run at 2am",
        }

        Returns:
            Dict mapping alternatives to their criterion scores and raw values
        """

        # Extract alternatives from scenario
        alternatives = []
        for alt_key in ['Alternative 1', 'Alternative 2', 'Alternative 3']:
            if alt_key in scenario and scenario[alt_key]:
                alternatives.append(scenario[alt_key])

        raw_results = {}

        for alt in alternatives:
            print(f"\nProcessing alternative: {alt}")

            # Parse alternative to extract run time and delay
            try:
                run_time_hour, delay_hours = self.parse_alternative(alt, scenario)
            except Exception as e:
                print(f"  ✗ Parsing ERROR: {e}")
                continue

            # Calculate raw criterion values
            try:
                energy_cost = self.calculate_energy_cost(
                    scenario['kwh/cycle'],
                    run_time_hour,
                    scenario['Peak Rate'],
                    scenario['Off-Peak Rate']
                )
            except Exception as e:
                print(f"  ✗ Energy cost ERROR: {e}")
                energy_cost = 0.0

            try:
                emissions = self.calculate_environmental_impact(scenario['kwh/cycle'])
            except Exception as e:
                print(f"  ✗ Emissions ERROR: {e}")
                emissions = 0.0

            try:
                comfort = self.calculate_comfort_score(
                    delay_hours,
                    run_time_hour,
                    scenario['Housing Type'],
                    scenario['Occupants'],
                    scenario['Appliance']
                )
            except Exception as e:
                print(f"  ✗ Comfort ERROR: {e}")
                comfort = 5.0

            try:
                practicality = self.calculate_practicality_score(
                    delay_hours,
                    run_time_hour,
                    scenario['Housing Type'],
                    scenario['Occupants'],
                    scenario['Appliance']
                )
            except Exception as e:
                print(f"  ✗ Practicality ERROR: {e}")
                practicality = 5.0

            raw_results[alt] = {
                'energy_cost_dollars': energy_cost,
                'emissions_lbs': emissions,
                'comfort_raw': comfort,
                'practicality_raw': practicality
            }

        # Apply value functions to get final 0-10 scores
        final_scores = {}

        for alt, raw in raw_results.items():
            print(f"\nApplying value functions for: {alt}")

            try:
                energy_vf = self.apply_value_function(
                    raw['energy_cost_dollars'],
                    self.VF_ENERGY_COST,
                    'energy_cost'
                )
                print(f"  After VF linear: Energy = {energy_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Energy VF ERROR: {e}")
                energy_vf = 5.0

            if 'Utility Budget' in scenario and scenario['Utility Budget'] > 0:
                # Convert per-cycle cost to monthly estimate (assume 30 cycles/month)
                monthly_cost = self.calculate_monthly_cost(
                    raw['energy_cost_dollars'],
                    cycles_per_month=30
                )

                budget_penalty = self.calculate_budget_penalty(
                    monthly_cost,
                    scenario['Utility Budget']
                )

                # Apply penalty to energy cost score
                energy_vf_penalized = energy_vf * budget_penalty

                print(f"  Budget check: ${monthly_cost:.2f}/month vs ${scenario['Utility Budget']:.2f} budget")
                print(
                    f"  Utilization: {monthly_cost / scenario['Utility Budget'] * 100:.1f}% : penalty: {budget_penalty:.3f}")
                print(f"  Energy score: {energy_vf:.2f} : {energy_vf_penalized:.2f} (after penalty)")

                energy_vf = energy_vf_penalized

            try:
                env_vf = self.apply_value_function(
                    raw['emissions_lbs'],
                    self.VF_ENVIRONMENTAL,
                    'environmental'
                )
                print(f"  After VF LinearL: Environmental = {env_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Environmental VF ERROR: {e}")
                env_vf = 5.0

            try:
                comfort_vf = self.apply_value_function(
                    raw['comfort_raw'],
                    self.VF_COMFORT,
                    'comfort'
                )
                print(f"  After VF logarithmic (a=1.5): Comfort = {comfort_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Comfort VF ERROR: {e}")
                comfort_vf = raw['comfort_raw']

            try:
                practicality_vf = self.apply_value_function(
                    raw['practicality_raw'],
                    self.VF_PRACTICALITY,
                    'practicality'
                )
                print(f"  After VF logarithmic (a=1.2): Practicality = {practicality_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Practicality VF ERROR: {e}")
                practicality_vf = raw['practicality_raw']

            final_scores[alt] = {
                'energy_cost_score': round(energy_vf, 2),
                'environmental_score': round(env_vf, 2),
                'comfort_score': round(comfort_vf, 2),
                'practicality_score': round(practicality_vf, 2),
                'raw_cost': round(raw['energy_cost_dollars'], 4),
                'raw_emissions': round(raw['emissions_lbs'], 3)
            }

            print(f"  : FINAL SCORES:")
            print(f"     Energy: {energy_vf:.2f}, Environmental: {env_vf:.2f}, "
                  f"Comfort: {comfort_vf:.2f}, Practicality: {practicality_vf:.2f}\n")

        return final_scores


def process_appliance_scenarios(csv_filename: str = "ApplianceScenarios.csv",
                                output_filename: str = "ground_truth_appliance.csv"):
    """
    Read Appliance scenarios from CSV and calculate ground truth scores for all alternatives.

    Args:
        csv_filename: Path to CSV file with scenarios
        output_filename: Where to save ground truth results

    Expected CSV columns:
        Description, Location, Utility Budget, Appliance, Housing Type,
        Occupants, Peak Rate, Off-Peak Rate, kwh/cycle, Appliance Age/Type,
        Alternative 1, Alternative 2, Alternative 3
    """

    df = pd.read_csv(csv_filename)

    print(f"Found {len(df)} appliance scenarios")

    calculator = ApplianceGroundTruthCalculator()

    results = []

    for idx, row in df.iterrows():
        print(f"\nProcessing scenario {idx + 1}/{len(df)}: {row['Appliance']} in {row['Location']}")

        # Collect alternatives
        alternatives = []
        for alt_col in ['Alternative 1', 'Alternative 2', 'Alternative 3']:
            alt_val = str(row[alt_col]).strip()

            if pd.isna(row[alt_col]) or alt_val == '' or alt_val == 'nan':
                continue
            alternatives.append(alt_val)
        scenario = {
            'Description': row['Description'],
            'Location': row['Location'],
            'Utility Budget': float(row['Utility Budget']),
            'Appliance': row['Appliance'],
            'Housing Type': row['Housing Type'],
            'Occupants': int(row['Occupants']),
            'Peak Rate': float(row['Peak Rate']),
            'Off-Peak Rate': float(row['Off-Peak Rate']),
            'kwh/cycle': float(row['kwh/cycle']),
            'Appliance Age/Type': row['Appliance Age/Type'],
            'Baseline Time': row['Baseline Time'],
            'Alternative 1': row['Alternative 1'],
            'Alternative 2': row['Alternative 2'],
            'Alternative 3': row['Alternative 3'],
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
                    'description': row['Description'],
                    'location': row['Location'],
                    'utility_budget': row['Utility Budget'],
                    'appliance': row['Appliance'],
                    'appliance_age_type': row['Appliance Age/Type'],
                    'housing_type': row['Housing Type'],
                    'occupants': row['Occupants'],
                    'kwh_per_cycle': row['kwh/cycle'],
                    'peak_rate': row['Peak Rate'],
                    'offpeak_rate': row['Off-Peak Rate'],
                    'alternative': alt,
                    'energy_cost_score': alt_scores['energy_cost_score'],
                    'environmental_score': alt_scores['environmental_score'],
                    'comfort_score': alt_scores['comfort_score'],
                    'practicality_score': alt_scores['practicality_score'],
                    'mavt_score': ranking_result["weighted_scores"][list(scores.keys()).index(alt)],
                    'rank': ranking_result["ranks"][list(scores.keys()).index(alt)],
                    'raw_cost': alt_scores['raw_cost'],
                    'raw_emissions': alt_scores['raw_emissions']
                }
                results.append(result_row)

        except Exception as e:
            print(f"ERROR processing scenario {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)

    print(f"\nGround truth saved to {output_filename}")
    print(f"Total alternatives scored: {len(results_df)}")
    return results_df

CRITERION_WEIGHTS = {
    "energy_cost": 0.30,
    "environmental": 0.35,
    "comfort": 0.20,
    "practicality": 0.15
}
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


if __name__ == "__main__":
    process_appliance_scenarios(
        csv_filename="../Scenario Files + Ground Truth/ApplianceScenarios.csv",
        output_filename="../Output Files/ground_truth_appliance.csv"
    )