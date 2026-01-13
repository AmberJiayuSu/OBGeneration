import pytest
import numpy as np

from generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions
from model.hvac import HVAC
from generator.hvac_generator import HVACAssumptions, HVACGenerator


class TestHVACGeneration:

    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_no_hvac(self, occ):
        """Test case where there is no HVAC system."""
        hvac_json = """
        {
            "heating": null,
            "cooling": null
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ1.household_annual_schedule()
        occupancy_mask,sleep_mask,_ = occ1.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is None
        assert cooling is None


    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_heating_no_control(self, occ):
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

        occ1 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ1.household_annual_schedule()
        occupancy_mask,sleep_mask,_ = occ1.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(occupancy_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(occupancy_mask[w]) for w in range(len(heating)))
        assert if_sublist_len_match
        if_same_values = all(all(h == HVACAssumptions.default().heating_defaults.active_setpoint for h in heating[w]) for w in range(len(heating)))
        assert if_same_values
        assert cooling is not None
        assert len(cooling) == len(occupancy_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(occupancy_mask[w]) for w in range(len(cooling)))
        assert if_sublist_len_match
        if_same_values = all(all(c == HVACAssumptions.default().cooling_defaults.active_setpoint for c in cooling[w]) for w in range(len(cooling)))
        assert if_same_values


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_binary_control(self,occ):
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

        occ2 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ2.household_annual_schedule()
        occupancy_mask,sleep_mask,active_mask = occ2.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(occupancy_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(occupancy_mask[w]) for w in range(len(heating)))
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
                             for h in range(len(heating[w])) if not occupancy_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off

        assert cooling is not None
        assert len(cooling) == len(occupancy_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(occupancy_mask[w]) for w in range(len(cooling)))
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
                             for h in range(len(cooling[w])) if not occupancy_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_off



    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_heating_valve_control(self,occ):
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

        occ2 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ2.household_annual_schedule()
        occupancy_mask,sleep_mask,active_mask = occ2.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(occupancy_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(occupancy_mask[w]) for w in range(len(heating)))
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
                             for h in range(len(heating[w])) if not occupancy_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_off
        assert cooling is None


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_setpoint_control(self,occ):
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
        occ2 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ2.household_annual_schedule()
        occupancy_mask,sleep_mask,active_mask = occ2.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(occupancy_mask)  
        if_sublist_len_match = all(len(heating[w]) == len(occupancy_mask[w]) for w in range(len(heating)))
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
                             for h in range(len(heating[w])) if not occupancy_mask[w][h]) 
                         for w in range(len(heating)))
        assert absent_setpoint

        assert cooling is not None
        assert len(cooling) == len(occupancy_mask)
        if_sublist_len_match = all(len(cooling[w]) == len(occupancy_mask[w]) for w in range(len(cooling)))
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
                             for h in range(len(cooling[w])) if not occupancy_mask[w][h]) 
                         for w in range(len(cooling)))
        assert absent_setpoint