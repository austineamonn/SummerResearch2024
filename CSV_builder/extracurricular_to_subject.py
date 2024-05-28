import pandas as pd

# Define the new mapping
subject_to_extracurricular = [
    ("Sports", "Medicine"),
    ("Music", "Music"),
    ("Art", "Art"),
    ("Volunteering", "Sociology"),
    ("Volunteering", "Psychology"),
    ("Volunteering", "Environmental Science"),
    ("Internship", "Business"),
    ("Internship", "Engineering"),
    ("Internship", "Medicine"),
    ("Debate Club", "Political Science"),
    ("Debate Club", "Law"),
    ("Drama Club", "Theater"),
    ("Drama Club", "Literature"),
    ("Science Club", "Biology"),
    ("Science Club", "Chemistry"),
    ("Science Club", "Physics"),
    ("Science Club", "Environmental Science"),
    ("Math Club", "Mathematics"),
    ("Math Club", "Computer Science"),
    ("Chess Club", "Mathematics"),
    ("Chess Club", "Computer Science"),
    ("Robotics Club", "Engineering"),
    ("Robotics Club", "Computer Science"),
    ("Environmental Club", "Environmental Science"),
    ("Environmental Club", "Biology"),
    ("Student Government", "Political Science"),
    ("Community Service", "Sociology"),
    ("Community Service", "Psychology"),
    ("Coding Club", "Computer Science"),
    ("Photography Club", "Art"),
    ("Photography Club", "Communications"),
    ("Dance Club", "Dance"),
    ("Dance Club", "Theater"),
    ("Foreign Language Club", "Languages"),
    ("Foreign Language Club", "Communications"),
    ("Literary Magazine", "Literature"),
    ("Literary Magazine", "Communications"),
    ("Yearbook Committee", "Communications"),
    ("Yearbook Committee", "Art")
]

# Convert to DataFrame
df_subject_to_extracurricular = pd.DataFrame(subject_to_extracurricular, columns=["Extracurricular", "Subject"])

# Save to CSV
df_subject_to_extracurricular.to_csv('extracurricular_to_subject.csv', index=False)
