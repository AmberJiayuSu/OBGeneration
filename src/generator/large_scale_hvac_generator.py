from stochastic.distribution import Distribution, CategoricalDistribution
from model.hvac import HVAC
from generator.hvac_generator import HVACGenerator, HVACAssumptions


## TODO: Not sure if this is the best way for aggregated HVAC representation
class HVACBaseRules:
    # probability of have the system on when active
    on_when_active: float
    # probability of have the system on when sleeping
    on_when_sleeping: float
    # probability of have the system on when away
    on_when_away: float
    # probability of the active setpoint from 10 to 32, 11 buckets in total, each bucket is 2 degrees
    active_setpoint: CategoricalDistribution
    # probability of the sleep setpoint from 10 to 32, 11 buckets in total, each bucket is 2 degrees
    sleep_setpoint: CategoricalDistribution
    # probability of the away setpoint from 0 to 32, 11 buckets in total, each bucket is 2 degrees
    away_setpoint: CategoricalDistribution


class HVACRules:
    hvac_assumptions: HVACAssumptions
    heating_rules: HVACBaseRules 
    cooling_rules: HVACBaseRules

    def __init__(self, hvac_assumptions: HVACAssumptions):
        self.hvac_assumptions = hvac_assumptions

    def aggregate_hvac(self, hvac_list: list[HVAC]):
        num_hvac = len(hvac_list)
        sum_on_when_active_heating = 0.0
        sum_on_when_sleeping_heating = 0.0
        sum_on_when_away_heating = 0.0
        active_setpoint_heating_probs = [0.0] * 11
        sleep_setpoint_heating_probs = [0.0] * 11
        away_setpoint_heating_probs = [0.0] * 11

        sum_on_when_active_cooling = 0.0
        sum_on_when_sleeping_cooling = 0.0
        sum_on_when_away_cooling = 0.0
        active_setpoint_cooling_probs = [0.0] * 11
        sleep_setpoint_cooling_probs = [0.0] * 11
        away_setpoint_cooling_probs = [0.0] * 11
        for hvac in hvac_list:
            if hvac.heating is not None:
                heating = hvac.heating
                if heating.type == "no_control":
                    sum_on_when_active_heating += 1.0
                    sum_on_when_sleeping_heating += 1.0
                    sum_on_when_away_heating += 1.0
                    setpoint_index = (self.hvac_assumptions.heating_defaults.active_setpoint - 10) // 2
                    active_setpoint_heating_probs[setpoint_index] += 1.0
                    sleep_setpoint_heating_probs[setpoint_index] += 1.0
                    away_setpoint_heating_probs[setpoint_index] += 1.0
                elif heating.type == "binary":
                    on_setpoint_index = (self.hvac_assumptions.heating_defaults.active_setpoint - 10) // 2
                    if heating.active_state:
                        sum_on_when_active_heating += 1.0
                        active_setpoint_heating_probs[on_setpoint_index] += 1.0
                    if heating.sleep_state:
                        sum_on_when_sleeping_heating += 1.0
                        sleep_setpoint_heating_probs[on_setpoint_index] += 1.0
                    if heating.absent_state:
                        sum_on_when_away_heating += 1.0
                        away_setpoint_heating_probs[on_setpoint_index] += 1.0
                elif heating.type == "valve":
                    if not heating.active_level == HVAC.IntensityLevel.MINIMUM:
                        sum_on_when_active_heating += 1.0
                        temp_ind = (HVACGenerator._get_trv_temp(self.hvac_assumptions, heating.active_level) - 10) // 2
                        active_setpoint_heating_probs[temp_ind] += 1.0
                    if not heating.sleep_level == HVAC.IntensityLevel.MINIMUM:
                        sum_on_when_sleeping_heating += 1.0
                        temp_ind = (HVACGenerator._get_trv_temp(self.hvac_assumptions, heating.sleep_level) - 10) // 2
                        sleep_setpoint_heating_probs[temp_ind] += 1.0
                    if not heating.absent_level == HVAC.IntensityLevel.MINIMUM:
                        sum_on_when_away_heating += 1.0
                        temp_ind = (HVACGenerator._get_trv_temp(self.hvac_assumptions, heating.absent_level) - 10) // 2
                        away_setpoint_heating_probs[temp_ind] += 1.0
                elif heating.type == "thermostat":
                    if heating.active_setpoint is not None:
                        sum_on_when_active_heating += 1.0
                        temp_ind = (heating.active_setpoint - 10) // 2
                        active_setpoint_heating_probs[temp_ind] += 1.0
                    if heating.sleep_setpoint is not None:
                        sum_on_when_sleeping_heating += 1.0
                        temp_ind = (heating.sleep_setpoint - 10) // 2
                        sleep_setpoint_heating_probs[temp_ind] += 1.0
                    if heating.absent_setpoint is not None:
                        sum_on_when_away_heating += 1.0
                        temp_ind = (heating.absent_setpoint - 10) // 2
                        away_setpoint_heating_probs[temp_ind] += 1.0
            if hvac.cooling is not None:
                cooling = hvac.cooling
                if cooling.type == "no_control":
                    sum_on_when_active_cooling += 1.0
                    sum_on_when_sleeping_cooling += 1.0
                    sum_on_when_away_cooling += 1.0
                    setpoint_index = (self.hvac_assumptions.cooling_defaults.active_setpoint - 20) // 2
                    active_setpoint_cooling_probs[setpoint_index] += 1.0
                    sleep_setpoint_cooling_probs[setpoint_index] += 1.0
                    away_setpoint_cooling_probs[setpoint_index] += 1.0
                elif cooling.type == "binary":
                    on_setpoint_index = (self.hvac_assumptions.cooling_defaults.active_setpoint - 20) // 2
                    if cooling.active_state:
                        sum_on_when_active_cooling += 1.0
                        active_setpoint_cooling_probs[on_setpoint_index] += 1.0
                    if cooling.sleep_state:
                        sum_on_when_sleeping_cooling += 1.0
                        sleep_setpoint_cooling_probs[on_setpoint_index] += 1.0
                    if cooling.absent_state:
                        sum_on_when_away_cooling += 1.0
                        away_setpoint_cooling_probs[on_setpoint_index] += 1.0
                elif cooling.type == "thermostat":
                    if cooling.active_setpoint is not None:
                        sum_on_when_active_cooling += 1.0
                        temp_ind = (cooling.active_setpoint - 20) // 2
                        active_setpoint_cooling_probs[temp_ind] += 1.0
                    if cooling.sleep_setpoint is not None:
                        sum_on_when_sleeping_cooling += 1.0
                        temp_ind = (cooling.sleep_setpoint - 20) // 2
                        sleep_setpoint_cooling_probs[temp_ind] += 1.0
                    if cooling.absent_setpoint is not None:
                        sum_on_when_away_cooling += 1.0
                        temp_ind = (cooling.absent_setpoint - 20) // 2
                        away_setpoint_cooling_probs[temp_ind] += 1.0
        active_heating_on_prob = sum_on_when_active_heating / num_hvac
        sleep_heating_on_prob = sum_on_when_sleeping_heating / num_hvac
        away_heating_on_prob = sum_on_when_away_heating / num_hvac
        sum_all_setpoint = sum(active_setpoint_heating_probs)
        active_cooling_on_prob = sum_on_when_active_cooling / num_hvac
        sleep_cooling_on_prob = sum_on_when_sleeping_cooling / num_hvac
        away_cooling_on_prob = sum_on_when_away_cooling / num_hvac


        self.heating_rules = HVACBaseRules(
            on_when_active=active_heating_on_prob,
            on_when_sleeping=sleep_heating_on_prob,
            on_when_away=away_heating_on_prob,
            active_setpoint=CategoricalDistribution(active_setpoint_heating_probs),
            sleep_setpoint=CategoricalDistribution(sleep_setpoint_heating_probs),
            away_setpoint=CategoricalDistribution(away_setpoint_heating_probs)
        )
        self.cooling_rules = HVACBaseRules(
            on_when_active=active_cooling_on_prob,
            on_when_sleeping=sleep_cooling_on_prob,
            on_when_away=away_cooling_on_prob,
            active_setpoint=CategoricalDistribution(active_setpoint_cooling_probs),
            sleep_setpoint=CategoricalDistribution(sleep_setpoint_cooling_probs),
            away_setpoint=CategoricalDistribution(away_setpoint_cooling_probs)
        )
                
                  