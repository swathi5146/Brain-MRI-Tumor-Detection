import os
import google.generativeai as genai
api_key=os.environ.get("GEMINI_API_KEY")

if not api_key:
	raise ValueError( "gEMINI_API_KEY not found. Please set it as an environment variable.")

genai.configure(api_key=api_key)

gemini_model=genai.GenerativeModel("gemini-3.1-flash-lite")

def generate_medical_report(predicted_class, confidence):
	prompt=f"""
    	You are assisting with an educational (non-diagnostic) brain MRI
    	classification project. The model predicted the class
    	"{predicted_class}" with a confidence of {confidence:.2f}%.
	
	 	Write a short, clear, patient-friendly report that includes:
    	1. A one-line explanation of what this class generally means.
    	2. 3-4 commonly associated symptoms (general educational info only).
    	3. A recommendation to consult a qualified radiologist/neurologist.

    	Keep the tone calm and educational. Do not present this as a
   	c	onfirmed diagnosis.
    	"""
	response= gemini_model.generate_content(prompt)
	return response.text
