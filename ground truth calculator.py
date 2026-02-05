import pandas as pd
import math
from typing import Dict, List, Tuple


class HVACGroundTruthCalculator:
    """
    Calculate physics-based ground truth scores for HVAC decision scenarios.
    Uses research from Ground Truth Data.pdf and ASHRAE standards.
    """

    EMISSIONS_FACTOR_PA = 0.6574
    ELECTRICITY_RATE_PA = 0.14
    SUMMER_COMFORT_RANGE = (73, 79)
    SUMMER_OPTIMAL = 76
    WINTER_COMFORT_RANGE = (68, 75)
    WINTER_OPTIMAL = 70

    # Linear VF for energy cost - equal marginal utility across range
    # Dyer & Sarin (1979): "For monetary attributes with small stakes relative to wealth,
    # linear utility is appropriate" (Management Science 26(8):810-822)
    # Newsham & Bowker (2010): TOU pricing shows linear elasticity regardless of starting
    # price level (Energy Policy 38:3289-3296)
    VF_ENERGY_COST = "linear"

    # Linear VF for environmental impact - physical units have linear marginal value
    # Kotchen & Moore (2007): "When environmental impacts are framed in absolute physical
    # units (tons CO₂, lbs emissions), people exhibit approximately linear preferences"
    # (J. Environmental Economics and Management 54(1):100-123)
    VF_ENVIRONMENTAL = "linear"
    VF_COMFORT = "logarithmic, a=1.5"
    VF_PRACTICALITY = "linear"

    def calculate_cooling_load(self, outdoor_temp: float, indoor_temp: float,
                               square_footage: int, r_value: int) -> float:
        """
        Calculate cooling load using ASHRAE cooling load temperature difference method.

        Citation: ASHRAE Handbook—Fundamentals (2021), Chapter 18
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
        print(f"  → Load calculated: {total_load:,.0f} BTU/hr")
        return max(0, total_load)

    def calculate_heating_load(self, outdoor_temp: float, indoor_temp: float,
                               square_footage: int, r_value: int) -> float:
        """
        Calculate heating load using ASHRAE heat loss method.

        Citation: ASHRAE Handbook—Fundamentals (2021), Chapter 18
        """
        delta_t = indoor_temp - outdoor_temp

        envelope_area = square_footage * 1.4

        u_factor = 1.0 / r_value

        conductive_loss = u_factor * envelope_area * delta_t

        internal_gains = 1000

        infiltration_loss = conductive_loss * 0.25

        total_load = conductive_loss + infiltration_loss - internal_gains
        print(f"  → Load calculated: {total_load:,.0f} BTU/hr")
        return max(0, total_load)

    def calculate_energy_consumption(self, load_btu_hr: float, seer: int,
                                     hvac_age: int, hours: float = 8) -> float:
        """
        Calculate energy consumption in kWh.

        Citations:
        - Huyen & Cetin (2019). Energies, 12(1):188
        - Age degradation: Alves et al. (2016). Energy and Buildings, 130:408-419

        Args:
            load_btu_hr: Cooling/heating load (BTU/hr)
            seer: SEER rating
            hvac_age: System age (years)
            hours: Operating hours

        Returns:
            Energy consumption in kWh
        """
        eer_estimated = seer * 0.875

        age_degradation_factor = 1 + (hvac_age * 0.01)

        adjusted_load = load_btu_hr * age_degradation_factor

        kw = (adjusted_load / eer_estimated) / 1000

        total_kwh = kw * hours
        print(f"  → Energy consumption: {total_kwh:.2f} kWh over {hours} hours")
        return total_kwh

    def calculate_comfort_score(self, indoor_temp: float, outdoor_temp: float,
                                household_size: int) -> float:
        """
        Calculate comfort score using ASHRAE 55 with adaptive comfort considerations.

        Citations:
        - Dear & Brager (2002). Energy and Buildings, 34:549-561 (adaptive comfort)
        - Wang & Hong (2020). Renewable & Sustainable Energy Reviews (occupant preferences)
        - Wu et al. (2018). Applied Energy, 212:577-591 (comfort ranges)
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
            # Wang & Hong (2020): "Observed acceptable temperature ranges span 7-12°C
            # (13-22°F), suggesting people tolerate wider ranges than ASHRAE 55 specifies"
            # Renewable & Sustainable Energy Reviews, DOI: 10.1016/j.rser.2019.109593
            comfort_score = 6 - (range_violation)

        if household_size > 3:
            size_penalty = (household_size - 3) * 0.3
            comfort_score -= size_penalty * (deviation / 3)

        return max(0, min(10, comfort_score))

    def calculate_practicality_score(self, outdoor_temp: float, indoor_temp: float,
                                     question_type: str = "simple") -> float:
        """
        Calculate practicality as likelihood of sustained behavioral adoption.
        NOT about comfort (that's the comfort criterion), but about behavioral abandonment.

        Citations:
        - Xu et al. (2017). "Investigating willingness to save energy and communication about
          energy use in the American workplace with the attitude-behavior-context model"
          Energy Research & Social Science 32:13-22
          Finding: Override behavior increases with extreme setpoints regardless of comfort

        - Stopps & Touchie (2021). "Residential smart thermostat use: An exploration of
          thermostat programming, environmental attitudes, and the influence of smart controls"
          Energy and Buildings 238:110834
          Finding: Complex schedules have 40-45% adoption rate vs 90%+ for simple setpoints

        - Karjalainen (2007). "Gender differences in thermal comfort and use of thermostats"
          Indoor Air 17(1):60-67
          Finding: Habituation difficulty for non-standard temperatures drives abandonment
        """

        if outdoor_temp > 75:  # Cooling mode
            if indoor_temp >= 82:
                extremity_penalty = (indoor_temp - 82) * 1.0  # Reduced from 1.2
            elif indoor_temp <= 71:
                extremity_penalty = (71 - indoor_temp) * 0.6  # Reduced from 0.8
            else:
                extremity_penalty = 0
        else:  # Heating mode
            if indoor_temp <= 63:
                # Softer penalty acknowledges that extreme setpoints may be necessary
                # for specific contexts (vacation, pipe freeze prevention)
                # Stopps & Touchie (2021): "Setback adoption varies by necessity context"
                extremity_penalty = (63 - indoor_temp) * 1.0  # Reduced from 1.5
            elif indoor_temp >= 76:
                extremity_penalty = (indoor_temp - 76) * 0.5  # Reduced from 0.7
            else:
                extremity_penalty = 0

        base_score = 10 - extremity_penalty

        # Apply minimum floor for practicaliy, because Even impractical alternatives have SOME non-zero likelihood
        base_score = max(1.5, base_score)
        # Component 2: Schedule complexity penalty
        # Stopps & Touchie (2021): Only 40-45% successfully maintain complex schedules
        # Therefore: complex penalty = 0.60 (vs 1.0 for simple)
        if question_type == "complex":
            base_score *= 0.60  # Changed from 0.85

        # Component 3: ΔT operational feasibility
        # Large ΔT indicates system operating at limits → lower reliability/higher failure risk
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

    def apply_value_function(self, raw_value: float, vf_spec: str, value_type: str) -> float:
        """
        Apply value function transformation to raw criterion values.

        Reference ranges derived from:
        - Huyen & Cetin (2019): Baseline consumption
        - Kim et al. (2024): Setpoint sensitivity
        - Cetin & Novoselac (2015): Runtime patterns
        - Alves et al. (2016): Degradation multipliers
        - Krarti & Howarth (2020): SEER-power relationships
        - EPA eGRID (2023): Grid emissions factors
        """
        reference_ranges = {
                'energy_cost': {
        # 5th-95th percentile from actual dataset distribution
        # Captures 90% of realistic alternatives, creates sensitivity in cluster region
        #
        # Min calculation:
        # Huyen & Cetin (2019): "Daily consumption of 6-8.2 kWh for well-insulated
        # homes with SEER 16+ under moderate conditions" (Energies 12(1):188)
        # → 8hr baseline: 2.0 kWh × $0.14/kWh = $0.28
        #
        # Kim et al. (2024): "Each 1°F increase in cooling setpoint reduces consumption
        # by 8-12%" (Building Simulation, DOI: 10.1007/s12273-024-1203-9)
        # → 82°F setpoint (6°F above 76°F): 48% reduction → $0.28 × 0.52 = $0.15
        #
        # Cetin & Novoselac (2015): "HVAC runtime shows significant variation based on
        # setpoint strategy and occupancy patterns" (Energy and Buildings 96:210-220)
        # → Accounting for partial operation: $0.47 (5th percentile from dataset)
        'min': 0.47,

        # Max calculation:
        # Alves et al. (2016): "Degraded systems (SEER 8-10) consume 2.5-4× more energy
        # than high-efficiency systems under identical loads" (Energy and Buildings 130:408-419)
        #
        # Krarti & Howarth (2020): "Low-efficiency systems (SEER 8-10) consume 3.8-4.5 kW
        # under design conditions" (J. Building Engineering 31:101457)
        # → 95th percentile from dataset: $3.31
        'max': 3.31,
        'decreasing': True
    },
    'environmental': {
        # Calculated from energy bounds using PA grid emissions factor
        #
        # EPA eGRID (2023): "Pennsylvania state-level CO₂ emission rate of 645.8 lbs
        # CO₂/MWh, or equivalently 0.6458 lbs CO₂/kWh" (eGRID2023 Summary Tables)
        #
        # Min: (0.47 / 0.14) × 8 hours × 0.6458 = 2.19 lbs CO₂
        # Max: (3.31 / 0.14) × 8 hours × 0.6458 = 15.45 lbs CO₂
        'min': 2.19,
        'max': 15.45,
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

        # REMOVED: x = max(x_min, min(x_max, raw_value))
        # Now use raw_value directly - allow extrapolation
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
                    u_x = 1.0  # Cap at perfect score
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
        question_type = "complex" if scenario.get('is_complex', False) else "simple"

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
                        print(f"  ⚠ Could not parse alternative: {alt}")
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
                scenario['hvac_age']
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
                question_type
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
                    scenario['vf_specs']['energy_cost'],
                    'energy_cost'
                )
                print(f"  After VF ({scenario['vf_specs']['energy_cost']}): Energy = {energy_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Energy VF ERROR: {e}")
                energy_vf = 5.0

            try:
                env_vf = self.apply_value_function(
                    raw['emissions_lbs'],
                    scenario['vf_specs']['environmental'],
                    'environmental'
                )
                print(f"  After VF ({scenario['vf_specs']['environmental']}): Environmental = {env_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Environmental VF ERROR: {e}")
                env_vf = 5.0

            try:
                comfort_vf = self.apply_value_function(
                    raw['comfort_raw'],
                    scenario['vf_specs']['comfort'],
                    'comfort'
                )
                print(f"  After VF ({scenario['vf_specs']['comfort']}): Comfort = {comfort_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Comfort VF ERROR: {e}")
                comfort_vf = raw['comfort_raw']

            try:
                practicality_vf = self.apply_value_function(
                    raw['practicality_raw'],
                    scenario['vf_specs']['practicality'],
                    'practicality'
                )
                print(f"  After VF ({scenario['vf_specs']['practicality']}): Practicality = {practicality_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Practicality VF ERROR: {e}")
                practicality_vf = raw['practicality_raw']

            final_scores[alt] = {
                'energy_cost_score': round(energy_vf, 2),
                'environmental_score': round(env_vf, 2),
                'comfort_score': round(comfort_vf, 2),
                'practicality_score': round(practicality_vf, 2),
                'raw_kwh': round(raw['kwh'], 2),
                'raw_cost': round(raw['energy_cost_dollars'], 2),
                'raw_emissions': round(raw['emissions_lbs'], 2)
            }

            print(f"  → FINAL SCORES:")
            print(
                f"     Energy: {energy_vf:.2f}, Environmental: {env_vf:.2f}, Comfort: {comfort_vf:.2f}, Practicality: {practicality_vf:.2f}\n")

        return final_scores


class ApplianceGroundTruthCalculator:
    """
    Calculate physics-based ground truth scores for appliance scheduling decisions.

    Key difference from HVAC: Energy consumption is constant per cycle,
    but cost varies by time-of-day due to TOU pricing.

    Citations:
    - Paetz et al. (2012): "Test-residents said to be able to postpone the dish-washer
      use by twelve hours without constraints in their daily routines"
      ACEEE Summer Study, Paper 0193-000232
    - EPA eGRID (2023): Pennsylvania grid emissions factor 0.6458 lbs CO2/kWh
    - PPL Electric (2024): TOU rate structure for southeastern PA
    - PECO Energy (2021): TOU rate structure for Philadelphia area
    """

    # Grid emissions factor (Pennsylvania)
    # EPA eGRID (2023): Pennsylvania state-level CO₂ emission rate
    EMISSIONS_FACTOR_PA = 0.6458  # lbs CO2/kWh

    # Fixed Peak Hours (2pm-6pm) - no seasonal variation
    # Based on PECO Energy standard TOU schedule
    PEAK_HOURS = (14, 18)  # 2 PM - 6 PM

    # Appliance Noise Levels
    # Whirlpool (2024): "Dishwashers typically have noise levels between 40 to 50 dBA"
    # Reviewed.com: Modern appliances operate at keyboard typing level (~45-50 dBA)
    NOISE_DISHWASHER = 45        # dBA typical modern dishwasher
    NOISE_WASHER = 50            # dBA typical washing machine
    NOISE_DRYER = 55             # dBA typical dryer

    # Apartment Noise Standards
    # All County Apartments: "Anything louder than 45 dB during daytime or 35 dB
    # in evening is deemed too loud"
    # Decibel Pro: "Time limits usually apply after 10 pm and until 7 am"
    NOISE_LIMIT_DAYTIME = 45     # dBA acceptable during day
    NOISE_LIMIT_EVENING = 35     # dBA acceptable after 10pm

    # Value Function Specifications
    # Linear VF for energy cost - equal marginal utility across range
    # Dyer & Sarin (1979): "For monetary attributes with small stakes relative to wealth,
    # linear utility is appropriate" (Management Science 26(8):810-822)
    VF_ENERGY_COST = "linear"

    # Linear VF for environmental impact - physical units have linear marginal value
    # Kotchen & Moore (2007): "When environmental impacts are framed in absolute physical
    # units (tons CO₂, lbs emissions), people exhibit approximately linear preferences"
    # (J. Environmental Economics and Management 54(1):100-123)
    VF_ENVIRONMENTAL = "linear"

    # Linear VF for comfort - start simple, test if needs adjustment
    VF_COMFORT = "linear"

    # Linear VF for practicality - adoption rates show approximately linear relationship
    VF_PRACTICALITY = "linear"

    def determine_rate_period(self, run_time_hour: int) -> str:
        """
        Determine if run time falls in peak or off-peak period.

        Args:
            run_time_hour: Hour of day (0-23, e.g., 19 for 7pm)

        Returns:
            "peak" or "offpeak"

        Citation: PECO Energy TOU schedule - 2pm-6pm peak hours
        """
        peak_start, peak_end = self.PEAK_HOURS

        if peak_start <= run_time_hour < peak_end:
            return "peak"

        return "offpeak"

    def calculate_energy_cost(self, kwh_cycle: float, run_time_hour: int,
                             peak_rate: float, offpeak_rate: float) -> float:
        """
        Calculate energy cost based on TOU rate structure.

        CRITICAL: Energy consumption (kWh) is FIXED - only COST varies by time.

        Args:
            kwh_cycle: Fixed energy per cycle (from Ground Truth Data.pdf Section 2)
            run_time_hour: When appliance runs (0-23)
            peak_rate: Peak period $/kWh
            offpeak_rate: Off-peak period $/kWh

        Returns:
            Energy cost in dollars

        Citations:
        - Ground Truth Data.pdf Section 2:
          Dishwasher 0.9-1.1 kWh/cycle (Energy Star), 1.4-2.0 kWh (old)
          Washer 0.15-0.25 kWh (front-load HE), 0.3-0.5 kWh (top-load)
          Dryer 0.8-1.5 kWh (heat pump), 2.5-4.0 kWh (standard electric)
        - Porras et al. (2020), Chen-Yu & Emmel (2018), Patel et al. (2021)
        """
        period = self.determine_rate_period(run_time_hour)

        if period == "peak":
            rate = peak_rate
        else:
            rate = offpeak_rate

        cost = kwh_cycle * rate
        print(f"  → Energy cost: {kwh_cycle} kWh × ${rate:.4f}/kWh ({period}) = ${cost:.4f}")
        return cost

    def calculate_environmental_impact(self, kwh_cycle: float) -> float:
        """
        Calculate CO2 emissions from electricity consumption.

        IMPORTANT: Environmental impact is SAME regardless of run time
        (using average grid emissions, not marginal hourly emissions).

        Args:
            kwh_cycle: Energy consumption per cycle

        Returns:
            CO2 emissions in pounds

        Citation: EPA eGRID (2023), Pennsylvania grid emissions factor
        0.6458 lbs CO2/kWh (state-level average)
        """
        emissions = kwh_cycle * self.EMISSIONS_FACTOR_PA
        print(f"  → Emissions: {kwh_cycle} kWh × {self.EMISSIONS_FACTOR_PA} lbs/kWh = {emissions:.3f} lbs CO2")
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

        Citations:
        - Paetz et al. (2012): "Test-residents said to be able to postpone the
          dish-washer use by twelve hours without constraints in their daily routines"
          ACEEE Summer Study, Paper 0193-000232

        - All County Apartments: "Anything louder than 45 dB during daytime or
          35 dB in evening is deemed too loud"

        - Whirlpool (2024): Modern dishwashers 40-50 dBA typical
          https://www.whirlpool.com/blog/kitchen/what-decibel-is-a-quiet-dishwasher.html
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

        print(f"  → Base comfort (delay={delay_hours}hr): {base_comfort}/10")

        # Component 2: Noise disruption penalty
        # Depends on: time of day + housing type + appliance noise
        noise_penalty = 0.0

        if appliance_type.lower() == "dishwasher":
            appliance_noise = self.NOISE_DISHWASHER
        elif appliance_type.lower() == "washer" or "washing" in appliance_type.lower():
            appliance_noise = self.NOISE_WASHER
        elif appliance_type.lower() == "dryer":
            appliance_noise = self.NOISE_DRYER
        else:
            appliance_noise = 50  # default

        # Late night running (10pm-7am)
        # Decibel Pro: "Time limits usually apply after 10 pm and until 7 am"
        if 22 <= run_time_hour or run_time_hour < 7:
            # Noise limit is 35 dBA in evening (All County Apartments)
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

                print(f"  → Noise penalty (late night, {housing_type}): -{noise_penalty:.1f}")

        # Component 3: Household size impact
        # Larger households → dishes pile up faster → delay worse
        # Ground Truth Data Section 9: Practicality varies by household context
        if occupants >= 5:
            size_penalty = 1.5
        elif occupants >= 3:
            size_penalty = 0.8
        else:
            size_penalty = 0.0

        # Apply size penalty proportional to delay
        # (No delay = no penalty, long delay = full penalty)
        size_penalty *= (delay_hours / 12.0)  # Scale by delay fraction
        print(f"  → Household size penalty ({occupants} occupants): -{size_penalty:.1f}")

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
        - Paetz et al. (2012): "Monetary savings became more important, which was
          also found to be one of the main underlying motives in shifting loads...
          During this test-living phase the residents saved around 6.5% on electricity costs"
          ACEEE Summer Study, Paper 0193-000232

        - Indonesia TOU study (2024): "63% of survey participants expect to opt into
          ToU scheme... 37% find ToU burdensome to implement and very demanding"
          PMC11190461

        - Shewale et al. (2023): TOU appliance scheduling adoption <20% without automation
          Arabian Journal for Science and Engineering, DOI: 10.1007/s13369-023-08178-w
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

        print(f"  → Base practicality (delay={delay_hours}hr): {base_practicality}/10")

        # Component 2: Timing complexity (remembering to run at specific time)
        # Late night/early morning = harder to remember/coordinate
        timing_penalty = 0.0

        # Paetz et al.: "If low-price zones applied on brink of day, it was
        # perceived as too early or too late"
        if 0 <= run_time_hour < 6:  # Middle of night (midnight-6am)
            timing_penalty = 2.0   # Very inconvenient timing
        elif 22 <= run_time_hour < 24:  # Late night (10pm-midnight)
            timing_penalty = 1.0   # Somewhat inconvenient

        print(f"  → Timing complexity penalty: -{timing_penalty:.1f}")

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
        print(f"  → Coordination penalty ({occupants} occupants): -{coordination_penalty:.1f}")

        final_practicality = base_practicality - timing_penalty - coordination_penalty

        # CRITICAL: Set minimum floor (even impractical has SOME adoption)
        # Following HVAC pattern: max(1.5, score)
        return max(1.5, min(10.0, final_practicality))

    def parse_alternative(self, alt: str, scenario: Dict) -> Tuple[int, float]:
        """
        Parse alternative text to extract run time and delay.

        SIMPLIFIED FORMAT - alternatives are just "Run at 7pm" (no extra description)

        Examples:
        - "Run at 7pm" → (19, 0)     # 7pm, no delay from 7pm baseline
        - "Run at 10pm" → (22, 3)    # 10pm, 3 hours after 7pm
        - "Run at 2am" → (2, 7)      # 2am, 7 hours after 7pm

        Args:
            alt: Alternative text string (e.g., "Run at 7pm")
            scenario: Full scenario dict (for fallback context)

        Returns:
            (run_time_hour, delay_hours)
        """
        import re

        # Extract time (e.g., "7pm", "10pm", "2am")
        time_match = re.search(r'(\d+)\s*(am|pm)', alt, re.IGNORECASE)
        if not time_match:
            print(f"  ⚠ Could not parse run time from: {alt}")
            # Default to immediate (no delay)
            return 19, 0.0  # 7pm, no delay

        hour = int(time_match.group(1))
        am_pm = time_match.group(2).lower()

        # Convert to 24-hour format
        if am_pm == "pm" and hour != 12:
            run_time_hour = hour + 12
        elif am_pm == "am" and hour == 12:
            run_time_hour = 0
        else:
            run_time_hour = hour

        # Calculate delay from baseline "after dinner" time (7pm = 19:00)
        baseline_hour = 19  # 7pm

        # Handle wraparound for late night (e.g., 2am is 7 hours after 7pm, not -17)
        if run_time_hour >= baseline_hour:
            delay_hours = float(run_time_hour - baseline_hour)
        else:
            # Crossed midnight: e.g., 7pm to 2am = 7 hours forward
            delay_hours = float(24 - baseline_hour + run_time_hour)

        print(f"  Parsed: '{alt}' → run at {run_time_hour:02d}:00, delay={delay_hours}hr")
        return run_time_hour, delay_hours

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
        # PLACEHOLDER - MUST BE DETERMINED FROM ACTUAL SCENARIOS
        # Process: Run all scenarios, analyze 5th-95th percentile, then set ranges
        reference_ranges = {
            'energy_cost': {
                # Will depend on: kWh range (0.9-4.0) × rate differential (0.09-0.18)
                # PLACEHOLDER values - update after data analysis
                'min': 0.08,
                'max': 0.65,
                'decreasing': True
            },
            'environmental': {
                # Calculated from energy bounds × emissions factor
                # PLACEHOLDER values - update after data analysis
                'min': 0.58,
                'max': 2.58,
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
                    u_x = 1.0  # Cap at perfect score
                else:
                    u_x = math.log(a * x_normalized + 1) / math.log(a + 1)

        else:
            u_x = x_normalized

        # Clamp final score to [0, 10]
        return max(0.0, min(10.0, u_x * 10.0))

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
            'vf_specs': {
                'energy_cost': 'linear',
                'environmental': 'linear',
                'comfort': 'linear',
                'practicality': 'linear'
            }
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
                    scenario['vf_specs']['energy_cost'],
                    'energy_cost'
                )
                print(f"  After VF ({scenario['vf_specs']['energy_cost']}): Energy = {energy_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Energy VF ERROR: {e}")
                energy_vf = 5.0

            try:
                env_vf = self.apply_value_function(
                    raw['emissions_lbs'],
                    scenario['vf_specs']['environmental'],
                    'environmental'
                )
                print(f"  After VF ({scenario['vf_specs']['environmental']}): Environmental = {env_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Environmental VF ERROR: {e}")
                env_vf = 5.0

            try:
                comfort_vf = self.apply_value_function(
                    raw['comfort_raw'],
                    scenario['vf_specs']['comfort'],
                    'comfort'
                )
                print(f"  After VF ({scenario['vf_specs']['comfort']}): Comfort = {comfort_vf:.2f}/10")
            except Exception as e:
                print(f"  ✗ Comfort VF ERROR: {e}")
                comfort_vf = raw['comfort_raw']

            try:
                practicality_vf = self.apply_value_function(
                    raw['practicality_raw'],
                    scenario['vf_specs']['practicality'],
                    'practicality'
                )
                print(f"  After VF ({scenario['vf_specs']['practicality']}): Practicality = {practicality_vf:.2f}/10")
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

            print(f"  → FINAL SCORES:")
            print(f"     Energy: {energy_vf:.2f}, Environmental: {env_vf:.2f}, "
                  f"Comfort: {comfort_vf:.2f}, Practicality: {practicality_vf:.2f}\n")

        return final_scores

class WaterGroundTruthCalculator:
    """
    PLACEHOLDER: Calculate ground truth for water heating decisions.

    Will need:
    - calculate_shower_energy(duration, water_heater_type, flow_rate)
    - calculate_comfort_score(duration, temperature)
    - apply_value_functions() (inherited or shared method)
    """
    pass


def process_hvac_scenarios(csv_filename: str = "Scenarios - HVAC Scenarios; GT (add complexity).csv",  output_filename: str = "ground_truth_hvac.csv"):
    """
    Read HVAC scenarios from CSV and calculate ground truth scores for all alternatives.

    Args:
        csv_filename: Path to CSV file with scenarios
        output_filename: Where to save ground truth results

    Expected CSV columns:
        Question, Location, Square Footage, Insulation, Household Size,
        Utility Budget, Housing Type, Outdoor Temp, House Age, R-Value,
        HVAC Age, SEER, Alternative 1, Alternative 2, Alternative 3,
        iscomplex
    """

    df = pd.read_csv(csv_filename)

    print(f"Found {len(df)} scenarios")

    calculator = HVACGroundTruthCalculator()

    results = []

    for idx, row in df.iterrows():
        print(f"Processing scenario {idx + 1}/{len(df)}: {row['Location']}")
        electricity_rate = 0.14

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
            'electricity_rate': electricity_rate,
            'is_complex': row['iscomplex'] == "TRUE",
            'alternatives': alternatives,
            'vf_specs': {
                'energy_cost': HVACGroundTruthCalculator.VF_ENERGY_COST,
                'environmental': HVACGroundTruthCalculator.VF_ENVIRONMENTAL,
                'comfort': HVACGroundTruthCalculator.VF_COMFORT,
                'practicality': HVACGroundTruthCalculator.VF_PRACTICALITY
            }
        }
        try:
            scores = calculator.calculate_scenario_scores(scenario)

            for alt, alt_scores in scores.items():
                result_row = {
                    'scenario_id': idx,
                    'question': row['Question'],
                    'location': row['Location'],
                    'outdoor_temp': row['Outdoor Temp'],
                    'electricity_rate': electricity_rate,
                    'alternative': alt,
                    'energy_cost_score': alt_scores['energy_cost_score'],
                    'environmental_score': alt_scores['environmental_score'],
                    'comfort_score': alt_scores['comfort_score'],
                    'practicality_score': alt_scores['practicality_score'],
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


if __name__ == "__main__":
    process_hvac_scenarios(
        csv_filename="Scenarios - HVAC Scenarios; GT (add complexity).csv",
        output_filename="ground_truth_hvac.csv"
    )

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)