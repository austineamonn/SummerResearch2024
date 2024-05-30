def get_combined_data():
    combined_dict = {
        'subjects_of_interest': [
            "Physics", "Mathematics", "Biology", "Chemistry", "History", "Literature", 
            "Computer Science", "Art", "Music", "Economics", "Psychology", "Sociology", 
            "Anthropology", "Political Science", "Philosophy", "Environmental Science", 
            "Geology", "Astronomy", "Engineering", "Medicine", "Law", "Business", 
            "Education", "Communications", "Languages", "Theater", "Dance"
        ],
        'course_to_subject': {
            'Physics': ['Physics'],
            'Mathematics': ['Mathematics'],
            'Biology': ['Biology'],
            'Chemistry': ['Chemistry'],
            'History': ['History'],
            'Literature': ['Literature', 'Poetry', 'Drama', 'Prose'],
            'Computer Science': ['Computer Science', 'Algorithms', 'Machine Learning'],
            'Art': ['Art', 'Painting', 'Sculpture', 'Photography'],
            'Music': ['Music', 'Jazz', 'Classical Music'],
            'Economics': ['Economics'],
            'Psychology': ['Psychology'],
            'Sociology': ['Sociology'],
            'Anthropology': ['Anthropology'],
            'Political Science': ['Political Science', 'International Relations', 'Comparative Politics'],
            'Philosophy': ['Philosophy', 'Ethics', 'Metaphysics', 'Epistemology', 'Logic'],
            'Environmental Science': ['Environmental Science', 'Climate Change', 'Conservation'],
            'Geology': ['Geology'],
            'Astronomy': ['Astronomy'],
            'Engineering': ['Engineering', 'Civil Engineering', 'Mechanical Engineering', 'Electrical Engineering', 'Chemical Engineering'],
            'Medicine': ['Medicine', 'Anatomy', 'Physiology', 'Pathology'],
            'Law': ['Law'],
            'Business': ['Business', 'Marketing', 'Finance', 'Management'],
            'Education': ['Education', 'Curriculum Development', 'Educational Psychology'],
            'Communications': ['Communications']
        },
        'extracurricular_list': [
            'Robotics Club',
            'Drama Club',
            'Debate Team',
            'Math Club',
            'Science Club',
            'Art Club',
            'Music Band',
            'Chess Club',
            'Environmental Club',
            'Volunteer Group',
            'Sports Team'
        ],
        'extracurricular_to_subject': {
            'Robotics Club': ['Engineering', 'Computer Science'],
            'Drama Club': ['Theater', 'Art'],
            'Debate Team': ['Political Science', 'Communications'],
            'Math Club': ['Mathematics'],
            'Science Club': ['Physics', 'Biology', 'Chemistry'],
            'Art Club': ['Art'],
            'Music Band': ['Music'],
            'Chess Club': ['Mathematics', 'Computer Science'],
            'Environmental Club': ['Environmental Science'],
            'Volunteer Group': ['Sociology', 'Psychology'],
            'Sports Team': ['Physical Education', 'Medicine']
        },
        'related_career_aspirations': {
            'Engineering': ['Engineer', 'Architect'],
            'Theater': ['Actor', 'Director'],
            'Political Science': ['Politician', 'Diplomat'],
            'Mathematics': ['Data Scientist', 'Mathematician'],
            'Physics': ['Physicist', 'Astronomer'],
            'Biology': ['Biologist', 'Geneticist'],
            'Chemistry': ['Chemist', 'Pharmacist'],
            'Art': ['Artist', 'Curator'],
            'Music': ['Musician', 'Composer'],
            'Sociology': ['Sociologist', 'Social Worker'],
            'Psychology': ['Psychologist', 'Therapist'],
            'Environmental Science': ['Environmental Scientist', 'Conservationist'],
            'Computer Science': ['Software Engineer', 'Data Scientist'],
            'Communications': ['Journalist', 'Public Relations Specialist']
        },
        'careers': [
            'Engineer', 'Architect', 'Actor', 'Director', 'Politician', 'Diplomat', 'Data Scientist', 'Mathematician',
            'Physicist', 'Astronomer', 'Biologist', 'Geneticist', 'Chemist', 'Pharmacist', 'Artist', 'Curator',
            'Musician', 'Composer', 'Sociologist', 'Social Worker', 'Psychologist', 'Therapist',
            'Environmental Scientist', 'Conservationist', 'Software Engineer', 'Data Scientist',
            'Journalist', 'Public Relations Specialist'
        ],
        'related_topics': {
            'Physics': ['Quantum Mechanics', 'Relativity'],
            'Mathematics': ['Calculus', 'Algebra'],
            'Biology': ['Genetics', 'Evolution'],
            'Chemistry': ['Organic Chemistry', 'Inorganic Chemistry'],
            'History': ['World War II', 'Ancient Civilizations'],
            'Literature': ['Poetry', 'Drama'],
            'Computer Science': ['Algorithms', 'Machine Learning'],
            'Art': ['Renaissance Art', 'Modern Art'],
            'Music': ['Classical Music', 'Jazz'],
            'Economics': ['Microeconomics', 'Macroeconomics'],
            'Psychology': ['Cognitive Psychology', 'Behavioral Psychology'],
            'Sociology': ['Urban Sociology', 'Rural Sociology'],
            'Anthropology': ['Cultural Anthropology', 'Physical Anthropology'],
            'Political Science': ['International Relations', 'Comparative Politics'],
            'Philosophy': ['Ethics', 'Metaphysics'],
            'Environmental Science': ['Climate Change', 'Conservation']
        },
        'course_type_to_learning_styles': {
            'Discussion/Recitation': ['Auditory', 'Read/Write'],
            'Lecture': ['Auditory', 'Read/Write'],
            'Lecture-Discussion': ['Auditory', 'Read/Write'],
            'Online': ['Read/Write'],
            'Independent Study': ['Read/Write'],
            'Laboratory': ['Kinesthetic'],
            'Online Lecture': ['Visual', 'Auditory'],
            'Online Discussion': ['Visual', 'Auditory'],
            'Conference': ['Auditory'],
            'Laboratory-Discussion': ['Kinesthetic'],
            'Online Lab': ['Kinesthetic'],
            'Study Abroad': ['Kinesthetic'],
            'Internship': ['Kinesthetic'],
            'Seminar': ['Auditory', 'Read/Write'],
            'Travel': ['Kinesthetic'],
            'Research': ['Read/Write'],
            'Studio': ['Visual'],
            'Packaged Section': ['Kinesthetic'],
            'Quiz': [],
            'Practice': ['Kinesthetic']
        },
        'race_ethnicity': {
            'European American or white': 53.4,
            'Latino/a/x American': 20.6,
            'African American or Black': 13.1,
            'Asian American': 7.6,
            'Multiracial': 4.3,
            'American Indian or Alaska Native': 0.7,
            'Pacific Islander': 0.3
        },
        'gender': {
            'Female': 58.36,
            'Male': 41.64
        },
        'international': {
            'Domestic': 94.08,
            'International': 5.92
        },
        'socioeconomic': {
            'Low': 20,
            'Middle': 60,
            'High': 20
        },
        'learning_style': {
            'Visual': 27.27,
            'Auditory': 23.56,
            'Read/Write': 21.16,
            'Kinesthetic': 28.01
        }
    }

    return combined_dict
