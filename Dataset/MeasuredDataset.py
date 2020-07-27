from Dataset.Dataset import MeasurementInfo, FBGMeasurementInfo

# 5 FBG

DATASET_5fbg_1 = FBGMeasurementInfo("Measured/5fbg/strain FBG1/").set_fbg_count(5)
DATASET_5fbg_1_1 = FBGMeasurementInfo("Measured/5fbg/strain FBG1/only FBG1").set_fbg_count(5)
DATASET_5fbg_2 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2/only FBG1(2 units)").set_fbg_count(5)
DATASET_5fbg_2_2 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2/only FBG2").set_fbg_count(5)
DATASET_5fbg_3 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3").set_fbg_count(5)
DATASET_5fbg_3_1 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG1(3 units)").set_fbg_count(5)
DATASET_5fbg_3_perfect = FBGMeasurementInfo("Measured/5fbg/perfect_3move").set_fbg_count(5)
DATASET_5fbg_3_2 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG2(2 units)").set_fbg_count(5)
DATASET_5fbg_3_3 = FBGMeasurementInfo("Measured/5fbg/strain FBG1 _ FBG2 _ FBG3/only FBG3").set_fbg_count(5)
DATASET_5fbg_1_perfect = FBGMeasurementInfo("Measured/5fbg/perfect5").set_fbg_count(5)

# 3 FBG

DATASET_3fbg_1 = FBGMeasurementInfo("Measured/3fbg/strain FBG1").set_fbg_count(3).set_threshold(1e-5)
DATASET_3fbg_1_2 = FBGMeasurementInfo("Measured/3fbg/strain FBG1-2").set_fbg_count(3)
DATASET_3fbg_perfect = FBGMeasurementInfo("Measured/3fbg/perfect").set_fbg_count(3)
DATASET_3fbg_2 = FBGMeasurementInfo("Measured/3fbg/2move").set_fbg_count(3)
DATASET_3fbg_2_noise = FBGMeasurementInfo("Measured/3fbg/2noisemove").set_fbg_count(3)

# 1 FBG

DATASET_1fbg_differentResolution = FBGMeasurementInfo("Measured/1fbg/DifferentResolution").set_fbg_count(1)

# background

DATASET_background = MeasurementInfo("Measured/background", 0)