import os
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def allowed_file(filename, allowed_extensions):
    """
    Check if a file has an allowed extension
    
    Args:
        filename (str): The name of the file
        allowed_extensions (set): A set of allowed file extensions
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_extension(filename):
    """
    Get the extension of a file
    
    Args:
        filename (str): The name of the file
        
    Returns:
        str: The file extension
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def get_result_label(prediction):
    """
    Convert the numeric prediction to a human-readable label
    
    Args:
        prediction (int or str): The model prediction (0 or 1)
        
    Returns:
        str: 'Leukemia' or 'Non-Leukemia'
    """
    # Convert to integer if it's a string
    if isinstance(prediction, str):
        try:
            prediction = int(prediction)
        except ValueError:
            # If it's already a label, return it
            if prediction.lower() in ['leukemia', 'non-leukemia']:
                return prediction.capitalize()
            return 'Unknown'
    
    # Convert numeric prediction to label
    if prediction == 1:
        return 'Leukemia'
    elif prediction == 0:
        return 'Non-Leukemia'
    else:
        return 'Unknown'

def get_result_message(prediction, confidence):
    """
    Get a message based on the prediction and confidence level
    
    Args:
        prediction (str): 'Leukemia' or 'Non-Leukemia'
        confidence (float): Confidence level (0-1)
        
    Returns:
        tuple: (message, alert_class) where alert_class is a Bootstrap alert class
    """
    if prediction.lower() == 'leukemia':
        if confidence > 0.9:
            return ("High probability of leukemia detected. Urgent medical consultation is recommended.", 
                   "danger")
        elif confidence > 0.7:
            return ("Moderate to high probability of leukemia detected. Medical consultation is recommended.", 
                   "warning")
        else:
            return ("Potential indicators of leukemia detected with low confidence. Further testing may be required.", 
                   "warning")
    else:
        if confidence > 0.9:
            return ("No indicators of leukemia detected with high confidence.", 
                   "success")
        elif confidence > 0.7:
            return ("No clear indicators of leukemia detected with moderate confidence.", 
                   "success")
        else:
            return ("No clear indicators of leukemia detected, but results have low confidence. Consider additional testing.", 
                   "info")
