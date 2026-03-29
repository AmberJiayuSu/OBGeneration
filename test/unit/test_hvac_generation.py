import pytest
import numpy as np

from ob_generation.generator.occupancy_generator import OccupancyGenerator, ClusterAssumptions
from ob_generation.model.hvac import HVAC
from ob_generation.generator.hvac_generator import HVACAssumptions, HVACGenerator
from ob_generation.model.occupancy import Occupancy

@pytest.fixture
def occ_1() -> Occupancy:
    occ_json ="""
        {
            "num_occupants": 3,
            "household_composition":{
                "occupants": [
                    {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                    {"weekday_cluster": "long_day_away", "weekend_cluster": "mostly_home"},
                    {"weekday_cluster": "morning_away", "weekend_cluster": "afternoon_away"}
                ]
            },
            "weekday_pattern": {
                "is_always_occupied": false,
                "away_interval": {"start_hour": 8, "end_hour": 14},
                "num_of_days":3,
                "away_time_rigidness": "somewhat_variable"
            },
            "weekend_pattern": {
                "is_always_occupied": false,
                "away_interval": {"start_hour": 16, "end_hour": 20},
                "away_time_rigidness": "somewhat_variable"
            },
            "sleep_pattern": {
                "is_always_awake": false,
                "sleep_time": {"start_hour": 22, "end_hour": 6},
                "sleep_time_rigidness": "mostly_consistent"
            }
        }"""
    return Occupancy.model_validate_json(occ_json)


class TestHVACGeneration:

    
    def test_no_hvac(self, occ_1):
        """Test case where there is no HVAC system."""
        hvac_json = """
        {
            "heating": null,
            "cooling": null
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.generate()
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)


        heating = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is None
        assert cooling is None



    def test_heating_no_control(self, occ_1):
        """Test case where there is heating but no control schedule."""
        hvac_json = """
        {
            "heating": {
                "type": "no_control"
            },
            "cooling": {
                "type": "no_control"
            }
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.generate()
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(active_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(active_mask[w]) for w in range(len(heating)))
        assert if_sublist_len_match
        if_same_values = all(all(h == HVACAssumptions.default().heating_defaults.active_setpoint for h in heating[w]) for w in range(len(heating)))
        assert if_same_values
        assert cooling is not None
        assert len(cooling) == len(active_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(active_mask[w]) for w in range(len(cooling)))
        assert if_sublist_len_match
        if_same_values = all(all(c == HVACAssumptions.default().cooling_defaults.active_setpoint for c in cooling[w]) for w in range(len(cooling)))
        assert if_same_values


    def test_binary_control(self,occ_1):
        hvac_json = """
        {
            "heating": {
                "type": "binary",
                "active_state": true,
                "sleep_state": true,
                "absent_state": false
            },
            "cooling": {
                "type": "binary",
                "active_state": true,
                "sleep_state": false,
                "absent_state": false
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.generate()
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(active_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(active_mask[w]) for w in range(len(heating)))
        assert if_sublist_len_match

        sleep_on = all(all(heating[w][h] == assumptions.heating_defaults.sleep_setpoint 
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_on

        active_on = all(all(heating[w][h] == assumptions.heating_defaults.active_setpoint 
                            for h in range(len(heating[w])) if active_mask[w][h] ) 
                        for w in range(len(heating)))
        assert active_on

        absent_off = all(all(heating[w][h] == assumptions.minimum_heating_setpoint
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off

        assert cooling is not None
        assert len(cooling) == len(active_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(active_mask[w]) for w in range(len(cooling)))
        assert if_sublist_len_match
        sleep_off = all(all(cooling[w][h] == assumptions.maximum_cooling_setpoint
                           for h in range(len(cooling[w])) if sleep_mask[w][h]) 
                       for w in range(len(cooling)))
        assert sleep_off
        active_on = all(all(cooling[w][h] == assumptions.cooling_defaults.active_setpoint 
                            for h in range(len(cooling[w])) if active_mask[w][h] ) 
                        for w in range(len(cooling)))
        assert active_on
        absent_off = all(all(cooling[w][h] == assumptions.maximum_cooling_setpoint
                             for h in range(len(cooling[w])) if not active_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_off




    def test_heating_valve_control(self,occ_1):
        hvac_json = """
        {
            "heating": {
                "type": "valve",
                "active_level": "high",
                "sleep_level": "medium",
                "absent_level": "low"
            },
            "cooling": null
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)

        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.generate()
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(active_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(active_mask[w]) for w in range(len(heating)))
        assert if_sublist_len_match

        sleep_on = all(all(heating[w][h] == assumptions.trv.valve_medium
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_on

        active_on = all(all(heating[w][h] == assumptions.trv.valve_high
                            for h in range(len(heating[w])) if active_mask[w][h] ) 
                        for w in range(len(heating)))
        assert active_on

        absent_off = all(all(heating[w][h] == assumptions.trv.valve_low
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off
        assert cooling is None



    def test_setpoint_control(self,occ_1):
        hvac_json = """
        {
            "heating": {
                "type": "thermostat",
                "active_setpoint": 21.0,
                "sleep_setpoint": 18.0,
                "absent_setpoint": 16.0
            },
            "cooling": {
                "type": "thermostat",
                "active_setpoint": 24.0,
                "sleep_setpoint": 27.0,
                "absent_setpoint": 30.0
            }
        }"""
        hvac = HVAC.model_validate_json(hvac_json)
        assumptions = HVACAssumptions.default()

        hvac_generator = HVACGenerator(hvac, assumptions)
       
        occ1 = OccupancyGenerator(occ_1, ClusterAssumptions.default(), 30)
        occ_sch = occ1.generate()
        active_mask,sleep_mask=OccupancyGenerator.active_sleep_mask(occ_sch)

        heating = hvac_generator.heating_setpoint_annual_schedule(active_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(active_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(active_mask)
        if_sublist_len_match = all(len(heating[w]) == len(active_mask[w]) for w in range(len(heating)))
        assert if_sublist_len_match
        sleep_setpoint = all(all(heating[w][h] == 18.0
                           for h in range(len(heating[w])) if sleep_mask[w][h]) 
                       for w in range(len(heating)))
        assert sleep_setpoint
        active_setpoint = all(all(heating[w][h] == 21.0
                            for h in range(len(heating[w])) if active_mask[w][h])
                        for w in range(len(heating)))
        assert active_setpoint
        absent_setpoint = all(all(heating[w][h] == 16.0
                             for h in range(len(heating[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_setpoint

        assert cooling is not None
        assert len(cooling) == len(active_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(active_mask[w]) for w in range(len(cooling)))
        assert if_sublist_len_match
        sleep_setpoint = all(all(cooling[w][h] == 27.0
                           for h in range(len(cooling[w])) if sleep_mask[w][h]) 
                       for w in range(len(cooling)))
        assert sleep_setpoint
        active_setpoint = all(all(cooling[w][h] == 24.0
                            for h in range(len(cooling[w])) if active_mask[w][h] )
                        for w in range(len(cooling)))
        assert active_setpoint
        absent_setpoint = all(all(cooling[w][h] == 30.0
                             for h in range(len(cooling[w])) if not active_mask[w][h] and not sleep_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_setpoint