import pandas as pd
def create_features(df):
    df = df.copy()
    df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]
    df["PopulationPerRoom"] = df["Population"] / df["AveRooms"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    return df