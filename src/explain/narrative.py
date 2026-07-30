from typing import List, Tuple

def get_readable_feature_name(feature_name: str) -> str:
    """
    Translates raw and engineered feature names into human-readable descriptions.
    """
    translations = {
        "TP2": "Compressor Pressure",
        "TP3": "Reservoir Pressure (pneumatic panel)",
        "H1": "Filter Pressure Drop",
        "DV_pressure": "Dryer Outlet Pressure",
        "Reservoirs": "Downstream Reservoirs Pressure",
        "Motor_current": "Motor Current draw",
        "Oil_temperature": "Compressor Oil Temperature",
        "COMP": "Air Intake Valve state",
        "DV_eletric": "Discharge Outlet Valve status",
        "TOWERS": "Dryer Tower status",
        "GPS_speed": "Train speed",
        "pressure_diff_tp3_tp2": "Reservoir-Compressor Pressure Difference",
        "pressure_diff_h1_dv": "Filter-Dryer Pressure Drop",
        "current_pressure_ratio": "Motor Current to Reservoir Pressure Ratio",
        "temp_pressure_ratio": "Oil Temperature to Reservoir Pressure Ratio",
        "is_compressing_under_load": "Compressor Operating Under Load",
        "is_compressor_off": "Compressor Inactive"
    }
    
    # Check exact match
    if feature_name in translations:
        return translations[feature_name]
        
    # Check engineered rolling features
    parts = feature_name.split("_")
    for raw_name, readable in translations.items():
        if feature_name.startswith(raw_name):
            suffix = feature_name[len(raw_name):]
            if "roll_mean" in suffix:
                window = suffix.split("_")[-1]
                return f"Average {readable} (past {window})"
            elif "roll_std" in suffix:
                window = suffix.split("_")[-1]
                return f"Fluctuations in {readable} (past {window})"
            elif "roll_min" in suffix:
                window = suffix.split("_")[-1]
                return f"Minimum {readable} (past {window})"
            elif "roll_max" in suffix:
                window = suffix.split("_")[-1]
                return f"Maximum {readable} (past {window})"
            elif "diff_mean" in suffix:
                window = suffix.split("_")[-1]
                return f"Rate of change in {readable} vs average (past {window})"
            elif "duty_cycle" in suffix:
                window = suffix.split("_")[-1]
                return f"Duty cycle frequency of {readable} (past {window})"
                
    return feature_name.replace("_", " ")

def generate_narrative_explanation(attributions: List[Tuple[str, float]], threshold_count: int = 3) -> str:
    """
    Translates SHAP attributions into a natural language narrative.
    Focuses on positive attributions (which push the probability towards failure).
    """
    # Filter for positive attributions (pushing toward anomaly warning)
    pos_attributions = [attr for attr in attributions if attr[1] > 0]
    
    if not pos_attributions:
        return "The APU operates within normal parameters. No significant indicators of degradation were found."
        
    # Take top N positive attributions
    top_pos = pos_attributions[:threshold_count]
    
    narrative_items = []
    for rank, (feat, val) in enumerate(top_pos, 1):
        readable_name = get_readable_feature_name(feat)
        narrative_items.append(f"{rank}. **{readable_name}** (attribution score: +{val:.4f})")
        
    intro = f"The APU warning alert was triggered because of the following primary degradation indicators:\n"
    body = "\n".join(narrative_items)
    
    conclusion = "\n\n**Action Recommendation:** Inspect these parameters on the compressor unit immediately to prevent service disruptions."
    return intro + body + conclusion

if __name__ == "__main__":
    # Test script
    mock_attributions = [
        ("Motor_current_roll_mean_30m", 0.12),
        ("Oil_temperature_diff_mean_60m", 0.08),
        ("TP3_roll_min_120m", 0.04),
        ("COMP_duty_cycle_30m", -0.05),
    ]
    print(generate_narrative_explanation(mock_attributions))
