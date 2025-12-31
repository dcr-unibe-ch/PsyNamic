import os
import unittest
from data.psynamic_dosage_norm import normalize_dosage, extract_dosages

class DosageNormTest(unittest.TestCase):
    
    def test_dosage_norm_unit_term(self):
        norm = '10 µg'
        dosages = [
            '10 mcg',
            '10 microg',
            '10 microgram'
        ]
        for dosage in dosages:
            self.assertEqual(normalize_dosage(dosage), norm)


        norm = '5 mg'
        dosage = '5mgs'
        self.assertEqual(normalize_dosage(dosage), norm)

        norm = '10 g'
        dosage = '10 grams'
        self.assertEqual(normalize_dosage(dosage), norm)

        norm = '3 kg'
        dosages = [
            '3 kilogram',
            '3 Kg'
        ]
        for dosage in dosages:
            self.assertEqual(normalize_dosage(dosage), norm)

        norm = '3 h'
        dosages = [
            '3 hours',
            '3 hour',
            '3 hr' 
        ]
        for dosage in dosages:
            self.assertEqual(normalize_dosage(dosage), norm)

        norm = '15 min'
        dosage = [
            '15 minutes',
            '15 minute',
            '15 mins' 
        ]
        for d in dosage:
            self.assertEqual(normalize_dosage(d), norm)

    def test_clean_dosage(self):
        input_output = {
            '0.5 mg / kg' : '0.5 mg/kg',
            'of 5 μg' : '5 μg',
            '100 mg / day' : '100 mg',
            '100 mg/daily' : '100 mg',
            '0.1-0.2 mg / kg / dose' : '0.1-0.2 mg/kg',
            '0.1-0.2 mg/kg/dose' : '0.1-0.2 mg/kg',
            '1 mg / kg of body weight' : '1 mg/kg',
            '0.25 mg / kg bw' : '0.25 mg/kg',
            '0.25 mg / kg /bw' : '0.25 mg/kg',
            '0.25 mg / kg / bw' : '0.25 mg/kg',
            '0.25 mg / kg / bodyweight' : '0.25 mg/kg',
            '0.25 mg / kg / bodyweight' : '0.25 mg/kg',
            '0.25 mg / kg / body-weight' : '0.25 mg/kg',
            '( 200 microg / kg )' : '200 µg/kg',
            '.5mg/kg' : '0.5 mg/kg',
            '2μg / kg / min' : '2 μg/kg/min',            
        }
        
        for inp, outp in input_output.items():
            self.assertEqual(normalize_dosage(inp), outp)


    def test_unit_spelling(self):
        input_output = {
            '0.15 mg kg(-1)' : '0.15 mg/kg',
            '0.15 mg kg(-1 )' : '0.15 mg/kg',
            '0.15 mg kg -1' : '0.15 mg/kg',
            '0.15 mg kg-1' : '0.15 mg/kg',
            '0.3 mg min(-1)' : '0.3 mg/min',
            '0.3 mg min -1' : '0.3 mg/min',
            '0.3 mg min-1' : '0.3 mg/min',
            '0.3 mg min (-1 )' : '0.3 mg/min',
            '1.4 microg kg(-1) min(-1 )' : '1.4 µg/kg/min',
            '0.25 mg hour(-1)' : '0.25 mg/h',
            '0.25 mg hour -1' : '0.25 mg/h',
            '0.25 mg hour-1' : '0.25 mg/h',
            '0.1 to 0.5 mg / Kg' : '0.1-0.5 mg/kg',
        }
        for inp, outp in input_output.items():
            self.assertEqual(normalize_dosage(inp), outp)

    def test_plus_minus(self):
        input_output = {
            '1,540 ± 920 mg' : '1540 mg',
            '1,540 +- 920 mg' : '1540 mg',
            '1540 ±920 mg' : '1540 mg',
            '1540+-920 mg' : '1540 mg',
        }
        for inp, outp in input_output.items():
            self.assertEqual(normalize_dosage(inp), outp)


class DosageExtractTest(unittest.TestCase):
    def test_simple_absolute_dose(self):
        # "10 mg"
        result = extract_dosages("10 mg")
        self.assertEqual(result["min"], 10)
        self.assertEqual(result["max"], 10)
        self.assertEqual(result["unit"], "mg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")

    def test_multiple_numeric_values_with_and(self):
        # "5 , 10 , and 20 µg"
        result = extract_dosages(normalize_dosage("5 , 10 , and 20 µg"))
        self.assertEqual(result["min"], 5)
        self.assertEqual(result["max"], 20)
        self.assertEqual(result["unit"], "µg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")

    def test_multiple_values_with_comma_no_whitespace(self):
        # "1,2,3 mg"
        result = extract_dosages("1,2,3 mg")
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 3)
        self.assertEqual(result["unit"], "mg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")

    def test_multiple_numeric_values_with_or(self):
        # "10 , 20 , 30 , or 40 mg"
        result = extract_dosages("10 , 20 , 30 , or 40 mg")
        self.assertEqual(result["min"], 10)
        self.assertEqual(result["max"], 40)
        self.assertEqual(result["unit"], "mg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")

    def test_relative_time_or(self):
        # "0.6 or 1 mg / min"
        result = extract_dosages("0.6 or 1 mg / min")
        self.assertEqual(result["min"], 0.6)
        self.assertEqual(result["max"], 1)
        self.assertEqual(result["unit"], "mg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertEqual(result["per_time_unit"], "min")
        self.assertEqual(result["dose_type"], "relative_time")

    def test_relative_weight_multiple_values(self):
        # "10 , 20 , 30 mg/70 kg"
        result = extract_dosages("10 , 20 , 30 mg/70 kg")
        self.assertEqual(result["min"], 10)
        self.assertEqual(result["max"], 30)
        self.assertEqual(result["unit"], "mg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 70)
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "relative_weight")

    def test_range_dose_dash(self):
        # "5 - 20 µg"
        result = extract_dosages("5 - 20 µg")
        self.assertEqual(result["min"], 5)
        self.assertEqual(result["max"], 20)
        self.assertEqual(result["unit"], "µg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")

    def test_range_with_to_and_weight(self):
        # "0.5 to 1.25 mg / kg"
        result = extract_dosages("0.5 to 1.25 mg / kg")
        self.assertEqual(result["min"], 0.5)
        self.assertEqual(result["max"], 1.25)
        self.assertEqual(result["unit"], "mg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 1)
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "relative_weight")

    def test_no_unit(self):
        # "100"
        result = extract_dosages("100")
        self.assertEqual(result["min"], 100)
        self.assertEqual(result["max"], 100)
        self.assertIsNone(result["unit"])
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")
    
    def test_starting_with_dot(self):
        # ".5 mg / kg"
        result = extract_dosages(".5 mg / kg")
        self.assertEqual(result["min"], 0.5)
        self.assertEqual(result["max"], 0.5)
        self.assertEqual(result["unit"], "mg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result['weight_reference'], 1)
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "relative_weight")

    def test_unit_spelling(self):
        # 0.25 mg kg(-1 ) hr(-1 )
        result = extract_dosages("0.25 mg kg(-1 ) hr(-1 )")
        self.assertEqual(result["min"], 0.25)
        self.assertEqual(result["max"], 0.25)
        self.assertEqual(result["unit"], "mg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 1)
        self.assertEqual(result["per_time_unit"], "h")
        self.assertEqual(result["dose_type"], "relative_weight_time")

    def test_unit_spelling_microg(self):
        # 1.4 microg kg(-1 ) min(-1 )
        result = extract_dosages("1.4 microg kg(-1 ) min(-1 )")
        self.assertEqual(result["min"], 1.4)
        self.assertEqual(result["max"], 1.4)
        self.assertEqual(result["unit"], "µg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 1)
        self.assertEqual(result["per_time_unit"], "min")

    def test_dosage_per_weight_per_time(self):
        # 2μg / kg / min
        result = extract_dosages("2μg / kg / min")
        self.assertEqual(result["min"], 2)
        self.assertEqual(result["max"], 2)
        self.assertEqual(result["unit"], "μg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 1)
        self.assertEqual(result["per_time_unit"], "min")   

    def test_range_with_weight_and_dose(self):
        # 0.1‐0.2 mg / kg / dose
        result = extract_dosages("0.1‐0.2 mg / kg / dose")
        self.assertEqual(result["min"], 0.1)
        self.assertEqual(result["max"], 0.2)
        self.assertEqual(result["unit"], "mg")
        self.assertEqual(result["per_weight_unit"], "kg")
        self.assertEqual(result["weight_reference"], 1)
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "relative_weight")

    def test_range_dash(self):
        # 5 - 20 µg
        result = extract_dosages("5 - 20 µg")
        self.assertEqual(result["min"], 5)
        self.assertEqual(result["max"], 20)
        self.assertEqual(result["unit"], "µg")
        self.assertIsNone(result["per_weight_unit"])
        self.assertIsNone(result["weight_reference"])
        self.assertIsNone(result["per_time_unit"])
        self.assertEqual(result["dose_type"], "absolute")   
