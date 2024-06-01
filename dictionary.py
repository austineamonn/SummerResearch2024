class Data:
    def __init__(self) -> None:
        pass
    def get_data(self):
        data_dict = {
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
        },
        'future_topics': [
            'Anthropology', 'Art', 'Biology', 'Chemistry',
            'Computer Science', 'Economics', 'Environmental Science',
            'History', 'Literature', 'Mathematics', 'Music', 'Philosophy',
            'Physics', 'Political Science', 'Psychology', 'Sociology'
        ]
    }
        return data_dict
    def get_data_generalization(self):
        data_generalization = {
            'race_ethnicity': {
                'full': {
                    'European American or white': 'Generalized',
                    'Latino/a/x American': 'Generalized',
                    'African American or Black': 'Generalized',
                    'Asian American': 'Generalized',
                    'Multiracial': 'Generalized',
                    'American Indian or Alaska Native': 'Generalized',
                    'Pacific Islander': 'Generalized'
                },
                'broad': {
                    'European American or white': 'Nonminority',
                    'Latino/a/x American': 'Minority',
                    'African American or Black': 'Minority',
                    'Asian American': 'Minority',
                    'Multiracial': 'Minority',
                    'American Indian or Alaska Native': 'Minority',
                    'Pacific Islander': 'Minority'
                },
                'slight': {
                    'European American or white': 'Nonminority',
                    'Latino/a/x American': 'Common Minority',
                    'African American or Black': 'Common Minority',
                    'Asian American': 'Common Minority',
                    'Multiracial': 'Uncommon Minority',
                    'American Indian or Alaska Native': 'Uncommon Minority',
                    'Pacific Islander': 'Uncommon Minority'
                },
                'none': lambda x: x  # None: No change, keep the list
            },
            'gender': {
                'full': {
                    'Female': 'Generalized',
                    'Male': 'Generalized'
                },
                'broad': {
                    'Female': 'Female',
                    'Male': 'Male'
                },
                'slight': {
                    'Female': 'Female',
                    'Male': 'Male'
                },
                'none': lambda x: x  # None: No change, keep the list
            },
            'international': {
                'full': {
                    'Domestic': 'Generalized',
                    'International': 'Generalized'
                },
                'broad': {
                    'Domestic': 'Domestic',
                    'International': 'International'
                },
                'slight': {
                    'Domestic': 'Domestic',
                    'International': 'International'
                },
                'none': lambda x: x  # None: No change, keep the list
            },
            'socioeconomic': {
                'full': {
                    'Low': 'Generalized',
                    'Middle': 'Generalized',
                    'High': 'Generalized'
                },
                'broad': {
                    'Low': 'Lower',
                    'Middle': 'Lower',
                    'High': 'Upper'
                },
                'slight': {
                    'Low': 'Low',
                    'Middle': 'Middle',
                    'High': 'High'
                },
                'none': lambda x: x  # None: No change, keep the list
            },
            'gpa': {
                'full': 0.0,
                'broad': lambda x: round(x, 1),  # Round to 1 decimal place for broad generalization
                'slight': lambda x: round(x, 2),  # Round to 2 decimal places for slight generalization
                'none': lambda x: x  # None: No change, keep the list
            },
            'student semester': {
                'full': 0.0,
                'broad': lambda x: x // 2 * 2,  # Broad: Group by every 2 years
                'slight': lambda x: x,  # Slight: No change, keep the year
                'none': lambda x: x  # None: No change, keep the list
            },
            'previous courses count': {
                'full': 0.0,
                'broad': lambda x: 0 if x <= 10 else (1 if x <= 20 else (2 if x <= 30 else (3 if x <= 40 else 4))),  # Broad: Group into ranges
                'slight': lambda x: 0 if x <= 5 else (1 if x <= 10 else (2 if x <= 15 else (3 if x <= 20 else (4 if x <= 25 else (5 if x <= 30 else (6 if x <= 35 else (7 if x <= 40 else (8 if x <= 45 else 9)))))))),  # Slight: Group into smaller ranges
                'none': lambda x: x  # None: No change, keep the list
            },
            'subjects of interest': {
                'full': ['Generalized'],
                'broad': lambda x: list(set([subject for subject in x if subject in self.get_data()['subjects_of_interest']])),  # Filter subjects
                'slight': lambda x: x,  # Slight: No change, keep the list
                'none': lambda x: x  # None: No change, keep the list
            },
            'career aspirations': {
                'full': ['Generalized'],
                'broad': lambda x: list(set([career for career in x])),  # Remove duplicates
                'slight': lambda x: x, # Slight: No change, keep the list
                'none': lambda x: x  # None: No change, keep the list
            },
            'extracurricular activities': {
                'full': ['Generalized'],
                'broad': lambda x: list(set([activity for activity in x])),  # Remove duplicates
                'slight': lambda x: x,  # Slight: No change, keep the list
                'none': lambda x: x  # None: No change, keep the list
            }
        }
        return data_generalization