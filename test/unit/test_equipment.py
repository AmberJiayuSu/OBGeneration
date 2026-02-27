import pytest
import numpy as np

from ob_generation.generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions
from ob_generation.model.equipment import Equipment
from ob_generation.generator.equipment_generator import EquipmentGenerator, EquipmentAssumptions, EventAssumptions
from ob_generation.model.occupant_profile import Occupant
from ob_generation.generator.ob_utils import ScheduleUtils
from ob_generation.generator.ob_generator import OccupantBehavior
from pathlib import Path

class TestEventAssumptions:
    """Test the EventAssumptions class for probability masking and resolution updates."""

    def test_masked_probabilities_normal(self):
        """Test masking with mixed active/inactive times and renormalization."""
        probabilities = [0.1, 0.2, 0.3, 0.4]
        event_assumptions = EventAssumptions(start_time_probabilities=probabilities)

        active_mask = [True, True, False, False]
        masked_probs = event_assumptions.get_masked_probabilities(active_mask)

        # Should zero out indices 2 and 3, then renormalize
        assert masked_probs[0] == pytest.approx(0.1 / 0.3)
        assert masked_probs[1] == pytest.approx(0.2 / 0.3)
        assert masked_probs[2] == 0.0
        assert masked_probs[3] == 0.0
        assert sum(masked_probs) == pytest.approx(1.0)

    def test_masked_probabilities_all_zeros_with_active(self):
        """Test when masking results in all zeros but some times are active."""
        probabilities = [0.0, 0.0, 0.1, 0.2]
        event_assumptions = EventAssumptions(start_time_probabilities=probabilities)

        active_mask = [True, True, False, False]
        masked_probs = event_assumptions.get_masked_probabilities(active_mask)

        # Should distribute uniformly over active times
        assert masked_probs[0] == 0.5
        assert masked_probs[1] == 0.5
        assert masked_probs[2] == 0.0
        assert masked_probs[3] == 0.0

    def test_masked_probabilities_never_occupied(self):
        """Test when never occupied (all mask False)."""
        probabilities = [0.1, 0.2, 0.3, 0.4]
        event_assumptions = EventAssumptions(start_time_probabilities=probabilities)

        active_mask = [False, False, False, False]
        masked_probs = event_assumptions.get_masked_probabilities(active_mask)

        # Should return all zeros
        assert all(p == 0.0 for p in masked_probs)


    def test_resolution_update(self):
        """Test aggregation from 5-minute to 15-minute resolution."""
        # 12 bins at 5-min resolution = 1 hour
        probabilities = [0.1] * 12
        event_assumptions = EventAssumptions(start_time_probabilities=probabilities)

        event_assumptions.update_resolution(15)

        # Should aggregate to 4 bins (15-min each)
        assert len(event_assumptions.start_time_probabilities) == 4
        assert all(p == pytest.approx(0.3) for p in event_assumptions.start_time_probabilities)



class TestEquipmentGenerator:
    """Test equipment schedule generation with various configurations."""


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_no_equipment(self, occ):
        """Test case where household has minimal equipment just baseload. """
        equipment_json = """
        {
            "laundry": {
                "has_washer": false,
                "has_dryer": false
            },
            "refrigerator": {
                "has_refrigerator": false
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)



        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        annual_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()


        # Should only have baseload
        assert len(annual_schedule) == 53
        assert all(len(week) > 0 for week in annual_schedule)
        # All laundry and dishwasher cycles should be 0
        if_laundry_zero = all(all(cycle == 0 for cycle in week) for week in laundry_cycles)
        assert if_laundry_zero
        if_dishwasher_zero = all(all(cycle == 0 for cycle in week) for week in dishwasher_cycles)
        assert if_dishwasher_zero
        # baseload      
        active_load = all(all(annual_schedule[w][h] == (assumptions.baseload +  assumptions.watts_per_person_active) for h in range(len(annual_schedule[w])) if active_mask[w][h]) for w in range(len(annual_schedule)))
        assert active_load
        sleep_load = all(all(annual_schedule[w][h] == (assumptions.baseload + assumptions.watts_per_person_sleep) for h in range(len(annual_schedule[w])) if sleep_mask[w][h]) for w in range(len(annual_schedule)))
        assert sleep_load
        absent_load = all(all(annual_schedule[w][h] == assumptions.baseload for h in range(len(annual_schedule[w])) if not active_mask[w][h] and not sleep_mask[w][h]) for w in range(len(annual_schedule)))
        assert absent_load


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_laundry_equipment_1(self, occ):
        """Test laundry schedule when laundry equipment exists."""
        equipment_json = """
        {
            "laundry": {
                "has_washer": true,
                "has_dryer": true,
                "washer_efficient": true,
                "dryer_efficient": true,
                "usage_frequency_per_week": {"min": 2, "max": 4}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        num_cycles = 0
        power = []
        for _ in range(52):
            laundry_schedule, cycles = equipment_gen.weekly_laundry_usage_schedule(active_mask[0],False)
            num_cycles += sum(cycles)
            power.append(sum(laundry_schedule)/4)


        assert 2*52 <= num_cycles <= 4*52
        assert pytest.approx(num_cycles / 52, rel=0.1) == 3.0
        power_mean = np.mean(power)
        expectation = (assumptions.laundry.efficient_washer.mean() * assumptions.laundry.washer_duration.mean() + assumptions.laundry.efficient_dryer.mean() * assumptions.laundry.dryer_duration.mean()) * 3 
        assert pytest.approx(power_mean, rel=0.1) == expectation


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_laundry_equipment_2(self, occ):
        """Test laundry schedule when laundry equipment exists."""
        equipment_json = """
        {
            "laundry": {
                "has_washer": true,
                "has_dryer": false,
                "washer_efficient": false,
                "usage_frequency_per_week": {"min": 1, "max": 1}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        num_cycles = 0
        power = []
        for _ in range(52):
            laundry_schedule, cycles = equipment_gen.weekly_laundry_usage_schedule(active_mask[0],False)
            num_cycles += sum(cycles)
            power.append(sum(laundry_schedule)/4)

        assert pytest.approx(num_cycles / 52, rel=0.1) == 1.0
        power_mean = np.mean(power)
        expectation = (assumptions.laundry.inefficient_washer.mean() * assumptions.laundry.washer_duration.mean() ) 
        assert pytest.approx(power_mean, rel=0.2) == expectation

    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_frige_equipment(self, occ):
        """Test refrigerator power for default assumptions."""
        equipment_json = """
        {
            "refrigerator": {
                "has_refrigerator": true,
                "efficient_refrigerator": true,
                "number_of_refrigerators": 1
            }
        }
        """
       
        assumptions = EquipmentAssumptions.default()
        equipment = Equipment.model_validate_json(equipment_json)
      
        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        refrigerator_assumptions = assumptions.refrigerator
        
        all_powers = []
        for _ in range(100):
            power = equipment_gen._get_fridge_power()
            all_powers.append(power)

        expected_power = refrigerator_assumptions.efficient_compact.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[0] + \
                         refrigerator_assumptions.efficient_small.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[1] + \
                         refrigerator_assumptions.efficient_medium.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[2] + \
                         refrigerator_assumptions.efficient_large.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[3]
        
        
        assert pytest.approx(expected_power, rel=0.1) == np.mean(all_powers)


        equipment_json = """
        {
            "refrigerator": {
                "has_refrigerator": true,
                "efficient_refrigerator": false,
                "number_of_refrigerators": 2
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)


        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        all_powers = []
        for _ in range(100):
            power = equipment_gen._get_fridge_power()
            all_powers.append(power)

        expected_power_1 = refrigerator_assumptions.inefficient_compact.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[0] + \
                         refrigerator_assumptions.inefficient_small.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[1] + \
                         refrigerator_assumptions.inefficient_medium.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[2] + \
                         refrigerator_assumptions.inefficient_large.mean() * refrigerator_assumptions.primary_refrigerator_size._probabilities[3]
        expected_power_2 = refrigerator_assumptions.inefficient_compact.mean() * refrigerator_assumptions.secondary_refrigerator_size._probabilities[0] + \
                            refrigerator_assumptions.inefficient_small.mean() * refrigerator_assumptions.secondary_refrigerator_size._probabilities[1] + \
                            refrigerator_assumptions.inefficient_medium.mean() * refrigerator_assumptions.secondary_refrigerator_size._probabilities[2] + \
                            refrigerator_assumptions.inefficient_large.mean() * refrigerator_assumptions.secondary_refrigerator_size._probabilities[3]
        
        expected_power = expected_power_1 + expected_power_2
        assert pytest.approx(np.mean(all_powers), rel=0.1) == expected_power


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_cooking_schedule(self, occ):
        """Test cooking schedule generation."""
        equipment_json = """
        {
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 10, "max": 15}
            }
        }
        """
        assumptions = EquipmentAssumptions.default()
        equipment = Equipment.model_validate_json(equipment_json)

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        cooking_assumptions = assumptions.cooking_products

        num_cooking = 0
        power = []
        for _ in range(52):
            cooking_schedule, end_times = equipment_gen.weekly_cooking_usage_schedule(active_mask[0],False)
            num_cooking += len(end_times)
            power.append(sum(cooking_schedule)/4)

        assert 10*52 <= num_cooking <= 15*52
        assert pytest.approx(num_cooking / 52, rel=0.1) == 12.5
        power_mean = np.mean(power)
        mean_time = cooking_assumptions.weekday_breakfast_duration.mean() * 5/21 + cooking_assumptions.weekend_breakfast_duration.mean() * 2/21 + \
        cooking_assumptions.weekday_lunch_duration.mean() * 5/21 + cooking_assumptions.weekend_lunch_duration.mean() * 2/21 + \
        cooking_assumptions.weekday_dinner_duration.mean() * 5/21 + cooking_assumptions.weekend_dinner_duration.mean() * 2/21
        expectation = cooking_assumptions.electric_cooking_products.mean() * mean_time * 12.5
        assert pytest.approx(power_mean, rel=0.2) == expectation


        

    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_dishwasher_after_each_meal(self, occ):
        """Test dishwasher running after each cooked meal."""
        equipment_json = """
        {
            "dishwasher": {
                "has_dishwasher": true,
                "efficient_dishwasher": true,
                "dishwashing_operational_logic": {
                    "pattern_type": "after_each_cooked_meal"
                }
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 5, "max": 7}
            }
        }
        """
        assumptions = EquipmentAssumptions.default()
        equipment = Equipment.model_validate_json(equipment_json)

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        dishwasher_assumptions = assumptions.dishwasher

        total_cycles = 0
        powers = []
        for _ in range(52):
            _, end_times = equipment_gen.weekly_cooking_usage_schedule(active_mask[0], False)
            dishwasher_schedule, cycles = equipment_gen.weekly_dishwasher_usage_schedule(
                active_mask[0], end_times, False
            )
            total_cycles += sum(cycles)
            powers.append(sum(dishwasher_schedule)/4)

        assert 5*52 <= total_cycles <= 7*52
        assert pytest.approx(total_cycles / 52, rel=0.1) == 6.0
        power_mean = np.mean(powers)
        expectation = dishwasher_assumptions.efficient_dishwasher.mean() * dishwasher_assumptions.dishwasher_cycle_duration.mean() * 6.0
        assert pytest.approx(power_mean, rel=0.2) == expectation


    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_dishwasher_daily_batch(self, occ):
        """Test dishwasher running once per day if cooking occurred."""
        equipment_json = """
        {
            "dishwasher": {
                "has_dishwasher": true,
                "dishwasher_efficient": true,
                "dishwashing_operational_logic": {
                    "pattern_type": "daily_batch_if_cooked"
                }
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 10, "max": 15}
            }
        }
        """
        assumptions = EquipmentAssumptions.default()
        equipment = Equipment.model_validate_json(equipment_json)

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        
        
        for _ in range(52):
            _, end_times = equipment_gen.weekly_cooking_usage_schedule(active_mask[0], False)
            dishwasher_schedule, cycles = equipment_gen.weekly_dishwasher_usage_schedule(
                active_mask[0], end_times, False
            )
            sum_cycles = sum(cycles)
            assert 1 <= sum_cycles <= 7  
            assert (sum(dishwasher_schedule) >0)



    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_dishwasher_independent_frequency(self, occ):
        """Test dishwasher with independent frequency pattern."""
        equipment_json = """
        {
            "dishwasher": {
                "has_dishwasher": true,
                "dishwasher_efficient": false,
                "dishwashing_operational_logic": {
                    "pattern_type": "independent_frequency",
                    "usage_frequency_per_week": {"min": 2, "max": 5}
                }
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 10, "max": 15}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        num_cycles = 0
        power = []
        for _ in range(52):
            _, end_times = equipment_gen.weekly_cooking_usage_schedule(active_mask[0], False)
            dishwasher_schedule, cycles = equipment_gen.weekly_dishwasher_usage_schedule(
                active_mask[0], end_times, False
            )
            num_cycles += sum(cycles)
            power.append(sum(dishwasher_schedule)/4)

        assert 2*52 <= num_cycles <= 5*52
        assert pytest.approx(num_cycles / 52, rel=0.1) == 3.5
        power_mean = np.mean(power)
        expectation = (assumptions.dishwasher.inefficient_dishwasher.mean() * assumptions.dishwasher.dishwasher_cycle_duration.mean()) * 3.5
        assert pytest.approx(power_mean, rel=0.2) == expectation
        

    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_dishwasher_wheneverfull(self, occ):
        equipment_json = """
        {
            "dishwasher": {
                "has_dishwasher": true,
                "dishwasher_efficient": false,
                "dishwashing_operational_logic": {
                    "pattern_type": "whenever_full"
                }
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 10, "max": 15}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        num_cycles = 0
        power = []
        
        for _ in range(52):
            _, end_times = equipment_gen.weekly_cooking_usage_schedule(active_mask[0], False)
            dishwasher_schedule, cycles = equipment_gen.weekly_dishwasher_usage_schedule(
                active_mask[0], end_times, False
            )
            num_cycles += sum(cycles)
            power.append(sum(dishwasher_schedule)/4)

        assert 3*52 <= num_cycles <= 5*52
        assert pytest.approx(num_cycles / 52, rel=0.1) == 4.16
        power_mean = np.mean(power)
        expectation = (assumptions.dishwasher.inefficient_dishwasher.mean() * assumptions.dishwasher.dishwasher_cycle_duration.mean()) * 4.16
        assert pytest.approx(power_mean, rel=0.2) == expectation
        

    

    @pytest.mark.parametrize("occ", ["occ_2"], indirect=True)
    def test_full_integration(self, occ):
        """Test full equipment generation with all equipment types."""
        equipment_json = """
        {
            "laundry": {
                "has_washer": true,
                "washer_efficient": true,
                "has_dryer": true,
                "dryer_efficient": false,
                "usage_frequency_per_week": {"min": 2, "max": 4}
            },
            "refrigeration": {
                "has_refrigerator": true,
                "efficient_refrigerator": true,
                "size": "large"
            },
            "dishwasher": {
                "has_dishwasher": true,
                "dishwasher_efficient": true,
                "dishwashing_operational_logic": {
                    "pattern_type": "daily_batch_if_cooked"
                }
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 12, "max": 18}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        occupancy_mask, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)

        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        annual_schedule, laundry_cycles, dishwasher_cycles = equipment_gen.equipment_annual_schedule()

        # Validate structure
        assert len(annual_schedule) == 53


        for week_schedule in annual_schedule:
            min_expected = assumptions.baseload
            assert all(power >= min_expected for power in week_schedule)
            assert len(set(week_schedule)) > 1


        assert all(sum(week) > 0 for week in laundry_cycles)
        assert all(sum(week) > 0 for week in dishwasher_cycles)



class TestAnnualConsumption:

    @pytest.mark.parametrize("occ_1", ["occ_1"], indirect=True)
    def test_laundry(self,occ_1):
        """Test annual laundry consumption calculation."""
        equipment_json = """
        {
            "laundry": {
                "has_washer": true,
                "washer_efficient": true,
                "has_dryer": true,
                "dryer_efficient": true,
                "usage_frequency_per_week": {"min": 1, "max": 4}
            },
            "refrigerator": {
                "has_refrigerator": false
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """
        equipment_json_2 = """
        {
            "laundry": {
                "has_washer": true,
                "washer_efficient": false,
                "has_dryer": true,
                "dryer_efficient": false,
                "usage_frequency_per_week": {"min": 1, "max": 4}
            },
            "refrigerator": {
                "has_refrigerator": false
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """

        equipment_1 = Equipment.model_validate_json(equipment_json)
        equipment_2 = Equipment.model_validate_json(equipment_json_2)

        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ_1, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)  

        equipment_gen_1 = EquipmentGenerator(
            equipment=equipment_1,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ_1.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        equipment_gen_2 = EquipmentGenerator(
            equipment=equipment_2,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ_1.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        laundry_annual_1 = equipment_gen_1.laundry_annual_schedule()
        laundry_annual_2 = equipment_gen_2.laundry_annual_schedule()

        total_consumption_1 = sum(sum(week) for week in laundry_annual_1) / 4000  # kWh
        total_consumption_2 = sum(sum(week) for week in laundry_annual_2) / 4000  # kWh
    
        assert total_consumption_2 > total_consumption_1
        assert total_consumption_1 > 300
        assert total_consumption_2 < 800


    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_refrigerator(self, occ):
        equipment_json = """
        {
            "laundry": {
                "has_washer": false
            },
            "refrigerator": {
                "has_refrigerator": true,
                "efficient_refrigerator": true,
                "number_of_refrigerators": 1
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """
        equipment_json_2 = """
        {
            "laundry": {
                "has_washer": false
            },
            "refrigerator": {
                "has_refrigerator": true,
                "efficient_refrigerator": false,
                "number_of_refrigerators": 2
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """

        equipment_1 = Equipment.model_validate_json(equipment_json)
        equipment_2 = Equipment.model_validate_json(equipment_json_2)

        assumptions = EquipmentAssumptions.default()

        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)  

        equipment_gen_1 = EquipmentGenerator(
            equipment=equipment_1,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        equipment_gen_2 = EquipmentGenerator(
            equipment=equipment_2,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )

        refrigerator_power_1 = []
        for _ in range(100):
            refrigerator_power = equipment_gen_1._get_fridge_power()
            annual_consumption = refrigerator_power * 24 * 365 / 1000  # kWh
            refrigerator_power_1.append(annual_consumption)

        refrigerator_power_2 = []
        for _ in range(100):
            refrigerator_power = equipment_gen_2._get_fridge_power()
            annual_consumption = refrigerator_power * 24 * 365 / 1000  # kWh
            refrigerator_power_2.append(annual_consumption)

        mean_1 = np.mean(refrigerator_power_1)
        mean_2 = np.mean(refrigerator_power_2)
        assert mean_1 > 400
        assert mean_1 < 700
        assert mean_2 > 1200
        assert mean_2 < 1800

    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_cooking(self, occ):
        equipment_json = """
        {
            "laundry": {
                "has_washer": false
            },
            "refrigerator": {
                "has_refrigerator": false
            },
            "dishwasher": {
                "has_dishwasher": false
            },
            "cooking_products": {
                "has_cooking_products": true,
                "cooking_products_fuel": "electric",
                "usage_frequency_per_week": {"min": 4, "max": 15}
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()
        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)
        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        cooking_schedule,_ = equipment_gen.cooking_annual_schedule()
        total_consumption = sum(sum(week) for week in cooking_schedule) / 4000  # kWh
        assert total_consumption > 200
        assert total_consumption < 400

    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_dishwasher(self, occ):
        equipment_json = """
        {
            "laundry": {
                "has_washer": false
            },
            "refrigerator": {
                "has_refrigerator": false
            },
            "dishwasher": {
                "has_dishwasher": true,
                "dishwasher_efficient": true,
                "dishwashing_operational_logic": {
                    "pattern_type": "independent_frequency",
                    "usage_frequency_per_week": {"min": 0, "max": 7}
                }
            },
            "cooking_products": {
                "has_cooking_products": false
            }
        }
        """
        equipment = Equipment.model_validate_json(equipment_json)
        assumptions = EquipmentAssumptions.default()
        occ_gen = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy, sleep = occ_gen.household_annual_schedule()
        _, sleep_mask, active_mask = occ_gen.get_annual_mask(occupancy, sleep)
        equipment_gen = EquipmentGenerator(
            equipment=equipment,
            active_mask=active_mask,
            sleep_mask=sleep_mask,
            occupancy=occupancy,
            num_occupants=occ.num_occupants,
            resolution_mins=15,
            equipment_assumptions=assumptions
        )
        _,cooking_end_times = equipment_gen.cooking_annual_schedule()
        dishwasher_schedule = equipment_gen.dishwasher_annual_schedule(cooking_end_times)
        total_consumption = sum(sum(week) for week in dishwasher_schedule) / 4000  # kWh
        assert total_consumption > 50
        assert total_consumption < 250


    @pytest.mark.parametrize("input_file", [
        "test/unit/input/OB_1.json",
        "test/unit/input/OB_2.json",
        "test/unit/input/OB_3.json"
    ])
    def test_equipment_annual_consumption(self, input_file):
        occupant = Occupant.from_json_file(Path(input_file))
        occupant_behavior = OccupantBehavior.to_OB_annual(15, occupant)
        equipment_schedule = occupant_behavior.equipment_schedule
        total_consumption = sum(equipment_schedule) / 4000  # kWh
        print(total_consumption)
        assert total_consumption > 3000