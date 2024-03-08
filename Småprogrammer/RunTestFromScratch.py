import subprocess

scripts = ["Småprogrammer\encode_question.py", "Småprogrammer\generateUserQ&A.py", "Småprogrammer\TestQuestions.py"]

for script in scripts:
    # Run each script one after another, waiting for each to complete before moving to the next
    subprocess.run(["python", script])