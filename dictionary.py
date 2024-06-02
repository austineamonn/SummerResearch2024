class Data:
    def __init__(self, config) -> None:
        self.config = config
        self.num_samples = config["synthetic_data"]["num_samples"]
        self.mutation_rate = config["privacy"]["mutation_rate"]
        self.epsilon = config["privacy"]["epsilon"]
        self.delta = config["privacy"]["delta"]
        self.noise_level = config["privacy"]["noise_level"]
        self.scale = config["privacy"]["scale"]
        self.shape = config["privacy"]["shape"]
        self.low = config["privacy"]["low"]
        self.high = config["privacy"]["high"]
        self.lam = config["privacy"]["lam"]
        self.salt_prob = config["privacy"]["salt_prob"]
        self.pepper_prob = config["privacy"]["pepper_prob"]
        self.variance = config["privacy"]["variance"]
        self.flip_prob = config["privacy"]["flip_prob"]
        self.snr = config["privacy"]["snr"]
        self.generalization_level = config["privacy"]["generalization_level"]
        self.sensitivity = config["privacy"]["sensitivity"]

    def get_data(self):
        data_dict = {
        'subjects_of_interest': [
            "Physics", "Mathematics", "Biology", "Chemistry", "History", "Literature",
            "Computer Science", "Art", "Music", "Economics", "Psychology", "Sociology",
            "Anthropology", "Political Science", "Philosophy", "Environmental Science",
            "Geology", "Astronomy", "Engineering", "Medicine", "Law", "Business",
            "Education", "Communications", "Languages", "Theater", "Dance", "Agricultural Sciences",
            "Urban Planning", "Nutrition", "Physical Education", "Kinesiology", "Landscape Architecture",
            "Jewish Studies", "Media and Cinema Studies", "Media", "Aerospace Engineering", 
            "African American Studies", "African Studies", "Agricultural Communications", 
            "Agricultural Education", "Agricultural Health and Safety", "Agricultural Science Education", 
            "Animal Sciences", "Arabic", "Archaeology", "Asian Studies", "Atmospheric Sciences", 
            "Basque", "Behavioral Sciences", "Biochemistry", "Biophysics", "Biostatistics", 
            "Biomedical Engineering", "Business Administration", "Comparative World Literature", 
            "Classical Civilizations", "Dance Theater", "East Asian Languages and Cultures", 
            "Electrical and Computer Engineering", "English", "Environmental Law", 
            "Environmental Policy and Planning", "Environmental Studies", "Family Health", 
            "Family Studies", "French", "Food Science and Human Nutrition", "Geographic Information Systems", 
            "German", "Global Studies", "Greek", "Gender and Women's Studies", "Health and Human Services", 
            "Hebrew", "Hindi", "Horticulture", "Human Development and Family Studies", "Integrative Biology", 
            "Italian", "Japanese", "Korean", "Latin", "Latin American and Caribbean Studies", 
            "Labor and Employment Relations", "Leadership Studies", "Linguistics", "Latina/Latino Studies", 
            "Management", "Marketing", "Microbiology", "Molecular and Cellular Biology", "Medieval Studies", 
            "Nuclear Engineering", "Near Eastern Languages and Cultures", "Neuroscience", 
            "Natural Resources and Environmental Sciences", "Pathobiology", "Persian", "Portuguese", 
            "Quechua", "Rehabilitation Education", "Religious Studies", "Recreation, Sport, and Tourism", 
            "Russian", "Scandinavian", "Spanish", "Special Education", "Statistics", "Swahili", 
            "Turkish", "Ukrainian", "Veterinary Medicine", "Women's Studies", "World Literatures", 
            "Yiddish", "Zulu"
        ],
        'course_to_subject': {
            # Asian American Studies
            "AAS": "Sociology",
            # Agricultural and Biological Engineering
            "ABE": "Agricultural Sciences",
            # Accountancy
            "ACCY": "Business",
            # Agricultural and Consumer Economics
            "ACE": "Economics",
            # Agricultural, Consumer, and Environmental Sciences
            "ACES": "Agricultural Sciences",
            # Advertising
            "ADV": "Communications",
            # Aerospace Engineering
            "AE": "Aerospace Engineering",
            # African Studies
            "AFAS": "African Studies",
            # African American Studies
            "AFRO": "African American Studies",
            # African Studies
            "AFST": "African Studies",
            # Agricultural Communications
            "AGCM": "Agricultural Communications",
            # Agricultural Education
            "AGED": "Agricultural Education",
            # Agricultural Health and Safety
            "AHS": "Agricultural Health and Safety",
            # Agricultural Science Education
            "AIS": "Agricultural Science Education",
            # Agricultural Leadership, Education, and Communications
            "ALEC": "Agricultural Sciences",
            # Animal Sciences
            "ANSC": "Animal Sciences",
            # Anthropology
            "ANTH": "Anthropology",
            # Arabic
            "ARAB": "Languages",
            # Architecture
            "ARCH": "Architecture",
            # Art
            "ART": "Art",
            # Art and Design
            "ARTD": "Art",
            # Art Education
            "ARTE": "Art",
            # Art Foundation
            "ARTF": "Art",
            # Art History
            "ARTH": "Art",
            # Art Jewelry/Metal
            "ARTJ": "Art",
            # Art Studio
            "ARTS": "Art",
            # Actuarial Science and Risk Management
            "ASRM": "Statistics",
            # Asian Studies
            "ASST": "Asian Studies",
            # Astronomy
            "ASTR": "Astronomy",
            # Atmospheric Sciences
            "ATMS": "Atmospheric Sciences",
            # Business Administration
            "BADM": "Business",
            # Basque
            "BASQ": "Languages",
            # Behavioral Sciences
            "BCOG": "Behavioral Sciences",
            # Bosnian, Croatian, Serbian
            "BCS": "Languages",
            # Business Data Innovation
            "BDI": "Business",
            # Biochemistry
            "BIOC": "Biochemistry",
            # Biomedical Engineering
            "BIOE": "Biomedical Engineering",
            # Biophysics
            "BIOP": "Biophysics",
            # Engineering
            "BSE": "Engineering",
            # Business and Technical Writing
            "BTW": "Communications",
            # Business
            "BUS": "Business",
            # Communication Arts and Sciences
            "CAS": "Communications",
            # Cell Biology
            "CB": "Biology",
            # Cell and Developmental Biology
            "CDB": "Biology",
            # Civil and Environmental Engineering
            "CEE": "Engineering",
            # Chemical and Biomolecular Engineering
            "CHBE": "Engineering",
            # Chemistry
            "CHEM": "Chemistry",
            # Chinese
            "CHIN": "Languages",
            # Public Health
            "CHLH": "Public Health",
            # Public Health
            "CHP": "Public Health",
            # Curriculum and Instruction
            "CI": "Education",
            # College of Education
            "CIC": "Education",
            # Classical Civilization
            "CLCV": "History",
            # Clinical Laboratory Science
            "CLE": "Languages",
            # Communication
            "CMN": "Communications",
            # Crop Sciences
            "CPSC": "Agricultural Sciences",
            # Computer Science
            "CS": "Computer Science",
            # Engineering
            "CSE": "Engineering",
            # Creative Writing
            "CW": "Literature",
            # Comparative World Literature
            "CWL": "Comparative World Literature",
            # Czech
            "CZCH": "Languages",
            # Dance
            "DANC": "Dance",
            # Dance Technology
            "DTX": "Dance",
            # East Asian Languages and Cultures
            "EALC": "Languages",
            # Electrical and Computer Engineering
            "ECE": "Engineering",
            # Economics
            "ECON": "Economics",
            # Education Practice
            "EDPR": "Education",
            # Education
            "EDUC": "Education",
            # Ecology and Evolutionary Biology
            "EEB": "Biology",
            # English as an International Language
            "EIL": "Languages",
            # English
            "ENG": "Literature",
            # English
            "ENGL": "Literature",
            # Environmental Sustainability
            "ENSU": "Environmental Science",
            # Entomology
            "ENT": "Agricultural Sciences",
            # Environmental Sciences
            "ENVS": "Environmental Science",
            # Education Policy, Organization and Leadership
            "EPOL": "Education",
            # Educational Psychology
            "EPSY": "Education",
            # Environmental Resource and Management
            "ERAM": "Agricultural Sciences",
            # Earth, Society, and Environment
            "ESE": "Environmental Science",
            # English as a Second Language
            "ESL": "Languages",
            # Engineering Technology and Management for Agricultural Systems
            "ETMA": "Engineering",
            # European Union Studies
            "EURO": "Languages",
            # Fine and Applied Arts
            "FAA": "Art",
            # Finance
            "FIN": "Business",
            # Foreign Language Teacher Education
            "FLTE": "Education",
            # French
            "FR": "Languages",
            # Food Science and Human Nutrition
            "FSHN": "Food Science and Human Nutrition",
            # Germanic Languages and Literatures
            "GC": "Languages",
            # Geology
            "GEOL": "Geology",
            # German
            "GER": "Languages",
            # Geography and Geographic Information Science
            "GGIS": "Geographic Information Systems",
            # Global Studies
            "GLBL": "Global Studies",
            # Greek and Modern Classical
            "GMC": "Languages",
            # Greek
            "GRK": "Languages",
            # Modern Greek
            "GRKM": "Languages",
            # General Studies
            "GS": "Gender and Women's Studies",
            # Germanic Studies
            "GSD": "Gender and Women's Studies",
            # Gender and Women's Studies
            "GWS": "Gender and Women's Studies",
            # Human Development and Family Studies
            "HDFS": "Human Development and Family Studies",
            # Hebrew
            "HEBR": "Languages",
            # History
            "HIST": "History",
            # Hindi
            "HNDI": "Languages",
            # Horticulture
            "HORT": "Agricultural Sciences",
            # Humanities Teaching
            "HT": "Languages",
            # Humanities
            "HUM": "Humanities",
            # Integrative Biology
            "IB": "Biology",
            # Industrial Engineering
            "IE": "Engineering",
            # Interdisciplinary Health Sciences
            "IHLT": "Health and Human Services",
            # Information Sciences
            "INFO": "Computer Science",
            # Information Sciences
            "IS": "Computer Science",
            # Italian
            "ITAL": "Languages",
            # Japanese
            "JAPN": "Languages",
            # Journalism
            "JOUR": "Communications",
            # Jewish Studies
            "JS": "Jewish Studies",
            # Kinesiology
            "KIN": "Kinesiology",
            # Korean
            "KOR": "Languages",
            # Landscape Architecture
            "LA": "Landscape Architecture",
            # Liberal Arts and Sciences
            "LAS": "Liberal Arts and Sciences",
            # Latin American and Caribbean Studies
            "LAST": "Latin American and Caribbean Studies",
            # Latin
            "LAT": "Languages",
            # Law
            "LAW": "Law",
            # Less Commonly Taught Languages
            "LCTL": "Languages",
            # Leadership Studies
            "LEAD": "Education",
            # Labor and Employment Relations
            "LER": "Labor and Employment Relations",
            # Linguistics
            "LING": "Linguistics",
            # Latina/Latino Studies
            "LLS": "Latina/Latino Studies",
            # Media and Cinema Studies
            "MACS": "Media and Cinema Studies",
            # Mathematics
            "MATH": "Mathematics",
            # Business
            "MBA": "Business",
            # Molecular and Cellular Biology
            "MCB": "Molecular and Cellular Biology",
            # Media
            "MDIA": "Media",
            # Medieval Studies
            "MDVL": "Medieval Studies",
            # Mechanical Engineering
            "ME": "Engineering",
            # Microbiology
            "MICR": "Microbiology",
            # Military Science
            "MILS": "Military Science",
            # Music Instrumental Performance
            "MIP": "Music",
            # Materials Science and Engineering
            "MSE": "Engineering",
            # Music
            "MUS": "Music",
            # Music
            "MUSC": "Music",
            # Music Education
            "MUSE": "Music",
            # Engineering
            "NE": "Engineering",
            # Neuroscience
            "NEUR": "Neuroscience",
            # Nuclear, Plasma, and Radiological Engineering
            "NPRE": "Engineering",
            # Nursing
            "NURS": "Nursing",
            # Nutrition
            "NUTR": "Nutrition",
            # Pathobiology
            "PATH": "Pathobiology",
            # Philosophy
            "PHIL": "Philosophy",
            # Physics
            "PHYS": "Physics",
            # Plant Biology
            "PLBIO": "Plant Biology",
            # Political Science
            "PLSC": "Political Science",
            # Portuguese
            "PORT": "Languages",
            # Psychology
            "PSYC": "Psychology",
            # Public Administration
            "PADM": "Public Administration",
            # Religion
            "RLST": "Religion",
            # Recreation, Sport and Tourism
            "RST": "Recreation, Sport and Tourism",
            # Russian, East European, and Eurasian Studies
            "REES": "Russian, East European, and Eurasian Studies",
            # Social Work
            "SOCW": "Social Work",
            # Sociology
            "SOC": "Sociology",
            # Spanish
            "SPAN": "Languages",
            # Speech and Hearing Science
            "SHS": "Speech and Hearing Science",
            # Statistics
            "STAT": "Statistics",
            # Sustainable Design
            "SD": "Sustainable Design",
            # Theatre
            "THEA": "Theatre",
            # Urban and Regional Planning
            "UP": "Urban and Regional Planning",
            # Veterinary Clinical Medicine
            "VCM": "Veterinary Clinical Medicine",
            # Veterinary Diagnostic and Production Animal Medicine
            "VDPAM": "Veterinary Diagnostic and Production Animal Medicine",
            # Veterinary Microbiology and Preventive Medicine
            "VMPM": "Veterinary Microbiology and Preventive Medicine",
            # Veterinary Pathology
            "VPTH": "Veterinary Pathology",
            # Veterinary Medicine
            "VETM": "Veterinary Medicine",
            # Women, Gender, and Sexuality Studies
            "WGSS": "Women, Gender, and Sexuality Studies",
            # African American Studies
            "AFAM": "African American Studies",
            # Architecture History
            "ARTHI": "Architecture History",
            # Art Studio
            "ARST": "Art Studio",
            # Art History
            "ARTH": "Art History",
            # Arts Management
            "ARTM": "Arts Management",
            # Atmospheric Science
            "ATMOS": "Atmospheric Science",
            # Biology
            "BIO": "Biology",
            # Chemistry
            "CHEM": "Chemistry",
            # Communications
            "COMM": "Communications",
            # Computer Engineering
            "COE": "Computer Engineering",
            # Creative Writing
            "CRW": "Creative Writing",
            # Design
            "DES": "Design",
            # Digital Media
            "DM": "Digital Media",
            # Earth Sciences
            "ESCI": "Earth Sciences",
            # Economics
            "ECON": "Economics",
            # Environmental Studies
            "ENVS": "Environmental Studies",
            # Film Studies
            "FILM": "Film Studies",
            # Fine Arts
            "FA": "Fine Arts",
            # French Studies
            "FRST": "French Studies",
            # Geography
            "GEOG": "Geography",
            # Geological Sciences
            "GSCI": "Geological Sciences",
            # German Studies
            "GERST": "German Studies",
            # Graphic Design
            "GD": "Graphic Design",
            # Health Education
            "HED": "Health Education",
            # History
            "HIST": "History",
            # International Relations
            "IR": "International Relations",
            # Italian Studies
            "ITST": "Italian Studies",
            # Japanese Studies
            "JPST": "Japanese Studies",
            # Latin American Studies
            "LAST": "Latin American Studies",
            # Linguistics
            "LING": "Linguistics",
            # Management
            "MGT": "Management",
            # Marketing
            "MKT": "Marketing",
            # Mathematics
            "MATH": "Mathematics",
            # Music Theory
            "MTH": "Music Theory",
            # Neuroscience
            "NEURO": "Neuroscience",
            # Nursing
            "NURS": "Nursing",
            # Physics
            "PHYS": "Physics",
            # Political Science
            "POLS": "Political Science",
            # Psychology
            "PSY": "Psychology",
            # Public Health
            "PUBH": "Public Health",
            # Sociology
            "SOC": "Sociology",
            # Spanish Studies
            "SPST": "Spanish Studies",
            # Statistics
            "STAT": "Statistics",
            # Theatre Arts
            "THA": "Theatre Arts",
            # Urban Planning
            "UP": "Urban Planning",
            # Women's Studies
            "WST": "Women's Studies"
        },
        'extracurricular_list': [
            "Robotics Club",
            "Drama Club",
            "Debate Team",
            "Math Club",
            "Science Club",
            "Art Club",
            "Music Band",
            "Chess Club",
            "Environmental Club",
            "Volunteer Group",
            "Sports Team",
            "Photography Club",
            "Literature Club",
            "History Club",
            "Language Club",
            "Computer Club",
            "Cooking Club",
            "Dance Team",
            "Film Club",
            "Journalism Club",
            "Astronomy Club",
            "Business Club",
            "Gardening Club",
            "Animal Rights Club",
            "Health and Fitness Club",
            "Model United Nations",
            "Entrepreneurship Club",
            "Book Club",
            "Coding Club",
            "Cultural Club",
            "Student Government",
            "Peer Tutoring",
            "Glee Club",
            "Scouting",
            "Yoga Club",
            "Martial Arts Club",
            "Speech Club",
            "Asian American Association",
            "Black Student Union",
            "Latino Student Union",
            "Gay-Straight Alliance",
            "Women’s Empowerment Club",
            "International Students Club",
            "Native American Club",
            "Disability Rights Club",
            "Economics Club",
            "Psychology Club",
            "Philosophy Club",
            "Geology Club",
            "Medicine Club",
            "Law Club",
            "Education Society",
            "Urban Planning Society",
            "Nutrition Club",
            "Kinesiology Club",
            "Landscape Architecture Society",
            "Jewish Studies Group",
            "Media and Cinema Studies Club",
            "Aerospace Engineering Society",
            "African American Studies Club",
            "African Studies Club",
            "Agricultural Communications Club",
            "Agricultural Education Society",
            "Agricultural Health and Safety Club",
            "Animal Sciences Club",
            "Arabic Language Club",
            "Archaeology Club",
            "Asian Studies Club",
            "Atmospheric Sciences Club",
            "Basque Culture Club",
            "Behavioral Sciences Club",
            "Biochemistry Club",
            "Biophysics Club",
            "Biostatistics Club",
            "Biomedical Engineering Society",
            "Business Administration Club",
            "Comparative World Literature Club",
            "Classical Civilizations Club",
            "Dance Theater Group",
            "East Asian Languages and Cultures Club",
            "Electrical and Computer Engineering Society",
            "English Club",
            "Environmental Law Society",
            "Environmental Policy and Planning Club",
            "Environmental Studies Club",
            "Family Health Club",
            "Family Studies Society",
            "French Club",
            "Food Science and Human Nutrition Society",
            "Geographic Information Systems Club",
            "German Club",
            "Global Studies Club",
            "Greek Language Club",
            "Gender and Women's Studies Club",
            "Health and Human Services Club",
            "Hebrew Language Club",
            "Hindi Language Club",
            "Horticulture Club",
            "Human Development and Family Studies Club",
            "Integrative Biology Society",
            "Italian Club",
            "Japanese Language Club",
            "Korean Language Club",
            "Latin Club",
            "Latin American and Caribbean Studies Club",
            "Labor and Employment Relations Society",
            "Leadership Studies Club",
            "Linguistics Club",
            "Latina/Latino Studies Club",
            "Management Club",
            "Marketing Club",
            "Microbiology Club",
            "Molecular and Cellular Biology Club",
            "Medieval Studies Club",
            "Nuclear Engineering Society",
            "Near Eastern Languages and Cultures Club",
            "Neuroscience Club",
            "Natural Resources and Environmental Sciences Club",
            "Pathobiology Club",
            "Persian Language Club",
            "Portuguese Language Club",
            "Quechua Language Club",
            "Rehabilitation Education Society",
            "Religious Studies Club",
            "Recreation, Sport, and Tourism Club",
            "Russian Language Club",
            "Scandinavian Studies Club",
            "Spanish Language Club",
            "Special Education Society",
            "Statistics Club",
            "Swahili Language Club",
            "Turkish Language Club",
            "Ukrainian Language Club",
            "Veterinary Medicine Society",
            "Women's Studies Club",
            "World Literatures Club",
            "Yiddish Language Club",
            "Zulu Language Club"
        ],
        'extracurricular_to_subject': {
            "Robotics Club": "Engineering",
            "Drama Club": "Theater",
            "Debate Team": "Communications",
            "Math Club": "Mathematics",
            "Science Club": ["Biology", "Chemistry", "Physics"],
            "Art Club": "Art",
            "Music Band": "Music",
            "Chess Club": "Mathematics",
            "Environmental Club": "Environmental Science",
            "Volunteer Group": "Sociology",
            "Sports Team": "Physical Education",
            "Photography Club": "Art",
            "Literature Club": "Literature",
            "History Club": "History",
            "Language Club": "Languages",
            "Computer Club": "Computer Science",
            "Cooking Club": "Home Economics",
            "Dance Team": "Dance",
            "Film Club": "Art",
            "Journalism Club": "Communications",
            "Astronomy Club": "Astronomy",
            "Business Club": "Business",
            "Gardening Club": "Environmental Science",
            "Animal Rights Club": "Sociology",
            "Health and Fitness Club": "Physical Education",
            "Model United Nations": "Political Science",
            "Entrepreneurship Club": "Business",
            "Book Club": "Literature",
            "Coding Club": "Computer Science",
            "Cultural Club": "Anthropology",
            "Student Government": "Political Science",
            "Peer Tutoring": "Education",
            "Glee Club": "Music",
            "Scouting": ["Environmental Science", "Sociology"],
            "Yoga Club": "Physical Education",
            "Martial Arts Club": "Physical Education",
            "Speech Club": "Communications",
            "Asian American Association": "Sociology",
            "Black Student Union": "Sociology",
            "Latino Student Union": "Sociology",
            "Gay-Straight Alliance": "Sociology",
            "Women’s Empowerment Club": "Sociology",
            "International Students Club": "Anthropology",
            "Native American Club": "Anthropology",
            "Disability Rights Club": "Sociology",
            "Economics Club": "Economics",
            "Psychology Club": "Psychology",
            "Philosophy Club": "Philosophy",
            "Geology Club": "Geology",
            "Medicine Club": "Medicine",
            "Law Club": "Law",
            "Education Society": "Education",
            "Urban Planning Society": "Urban Planning",
            "Nutrition Club": "Nutrition",
            "Kinesiology Club": "Kinesiology",
            "Landscape Architecture Society": "Landscape Architecture",
            "Jewish Studies Group": "Jewish Studies",
            "Media and Cinema Studies Club": "Media and Cinema Studies",
            "Aerospace Engineering Society": "Aerospace Engineering",
            "African American Studies Club": "African American Studies",
            "African Studies Club": "African Studies",
            "Agricultural Communications Club": "Agricultural Communications",
            "Agricultural Education Society": "Agricultural Education",
            "Agricultural Health and Safety Club": "Agricultural Health and Safety",
            "Animal Sciences Club": "Animal Sciences",
            "Arabic Language Club": "Arabic",
            "Archaeology Club": "Archaeology",
            "Asian Studies Club": "Asian Studies",
            "Atmospheric Sciences Club": "Atmospheric Sciences",
            "Basque Culture Club": "Basque",
            "Behavioral Sciences Club": "Behavioral Sciences",
            "Biochemistry Club": "Biochemistry",
            "Biophysics Club": "Biophysics",
            "Biostatistics Club": "Biostatistics",
            "Biomedical Engineering Society": "Biomedical Engineering",
            "Business Administration Club": "Business Administration",
            "Comparative World Literature Club": "Comparative World Literature",
            "Classical Civilizations Club": "Classical Civilizations",
            "Dance Theater Group": "Dance Theater",
            "East Asian Languages and Cultures Club": "East Asian Languages and Cultures",
            "Electrical and Computer Engineering Society": "Electrical and Computer Engineering",
            "English Club": "English",
            "Environmental Law Society": "Environmental Law",
            "Environmental Policy and Planning Club": "Environmental Policy and Planning",
            "Environmental Studies Club": "Environmental Studies",
            "Family Health Club": "Family Health",
            "Family Studies Society": "Family Studies",
            "French Club": "French",
            "Food Science and Human Nutrition Society": "Food Science and Human Nutrition",
            "Geographic Information Systems Club": "Geographic Information Systems",
            "German Club": "German",
            "Global Studies Club": "Global Studies",
            "Greek Language Club": "Greek",
            "Gender and Women's Studies Club": "Gender and Women's Studies",
            "Health and Human Services Club": "Health and Human Services",
            "Hebrew Language Club": "Hebrew",
            "Hindi Language Club": "Hindi",
            "Horticulture Club": "Horticulture",
            "Human Development and Family Studies Club": "Human Development and Family Studies",
            "Integrative Biology Society": "Integrative Biology",
            "Italian Club": "Italian",
            "Japanese Language Club": "Japanese",
            "Korean Language Club": "Korean",
            "Latin Club": "Latin",
            "Latin American and Caribbean Studies Club": "Latin American and Caribbean Studies",
            "Labor and Employment Relations Society": "Labor and Employment Relations",
            "Leadership Studies Club": "Leadership Studies",
            "Linguistics Club": "Linguistics",
            "Latina/Latino Studies Club": "Latina/Latino Studies",
            "Management Club": "Management",
            "Marketing Club": "Marketing",
            "Microbiology Club": "Microbiology",
            "Molecular and Cellular Biology Club": "Molecular and Cellular Biology",
            "Medieval Studies Club": "Medieval Studies",
            "Nuclear Engineering Society": "Nuclear Engineering",
            "Near Eastern Languages and Cultures Club": "Near Eastern Languages and Cultures",
            "Neuroscience Club": "Neuroscience",
            "Natural Resources and Environmental Sciences Club": "Natural Resources and Environmental Sciences",
            "Pathobiology Club": "Pathobiology",
            "Persian Language Club": "Persian",
            "Portuguese Language Club": "Portuguese",
            "Quechua Language Club": "Quechua",
            "Rehabilitation Education Society": "Rehabilitation Education",
            "Religious Studies Club": "Religious Studies",
            "Recreation, Sport, and Tourism Club": "Recreation, Sport, and Tourism",
            "Russian Language Club": "Russian",
            "Scandinavian Studies Club": "Scandinavian",
            "Spanish Language Club": "Spanish",
            "Special Education Society": "Special Education",
            "Statistics Club": "Statistics",
            "Swahili Language Club": "Swahili",
            "Turkish Language Club": "Turkish",
            "Ukrainian Language Club": "Ukrainian",
            "Veterinary Medicine Society": "Veterinary Medicine",
            "Women's Studies Club": "Women's Studies",
            "World Literatures Club": "World Literatures",
            "Yiddish Language Club": "Yiddish",
            "Zulu Language Club": "Zulu"
        },
        'related_career_aspirations': {
            "Physics": ["Physicist", "Research Scientist", "Data Analyst", "Engineer", "Professor", "Astrophysicist"],
            "Mathematics": ["Mathematician", "Statistician", "Actuary", "Data Scientist", "Operations Research Analyst", "Financial Analyst"],
            "Biology": ["Biologist", "Research Scientist", "Microbiologist", "Geneticist", "Biotechnologist", "Environmental Scientist"],
            "Chemistry": ["Chemist", "Pharmacologist", "Forensic Scientist", "Chemical Engineer", "Toxicologist", "Materials Scientist"],
            "History": ["Historian", "Archivist", "Museum Curator", "Teacher", "Researcher", "Historical Consultant"],
            "Literature": ["Writer", "Editor", "Publisher", "Teacher", "Journalist", "Literary Critic"],
            "Computer Science": ["Software Developer", "Data Scientist", "Cybersecurity Specialist", "Systems Analyst", "AI Engineer", "Database Administrator"],
            "Art": ["Artist", "Graphic Designer", "Art Director", "Museum Curator", "Art Teacher", "Illustrator"],
            "Music": ["Musician", "Music Teacher", "Composer", "Sound Engineer", "Music Therapist", "Conductor"],
            "Economics": ["Economist", "Financial Analyst", "Policy Analyst", "Consultant", "Statistician", "Market Research Analyst"],
            "Psychology": ["Psychologist", "Therapist", "Counselor", "Human Resources Specialist", "Researcher", "Clinical Psychologist"],
            "Sociology": ["Sociologist", "Social Worker", "Policy Analyst", "Urban Planner", "Market Research Analyst", "Community Service Manager"],
            "Anthropology": ["Anthropologist", "Archaeologist", "Cultural Resource Manager", "Museum Curator", "Researcher", "Forensic Anthropologist"],
            "Political Science": ["Politician", "Policy Analyst", "Public Relations Specialist", "Diplomat", "Political Consultant", "Lobbyist"],
            "Philosophy": ["Philosopher", "Ethicist", "Professor", "Writer", "Researcher", "Policy Analyst"],
            "Environmental Science": ["Environmental Scientist", "Ecologist", "Conservationist", "Environmental Consultant", "Wildlife Biologist", "Environmental Engineer"],
            "Geology": ["Geologist", "Hydrologist", "Environmental Consultant", "Mining Engineer", "Seismologist", "Petroleum Geologist"],
            "Astronomy": ["Astronomer", "Astrophysicist", "Research Scientist", "Professor", "Data Analyst", "Planetarium Director"],
            "Engineering": ["Civil Engineer", "Mechanical Engineer", "Electrical Engineer", "Software Engineer", "Aerospace Engineer", "Biomedical Engineer"],
            "Medicine": ["Doctor", "Surgeon", "Nurse", "Medical Researcher", "Pharmacist", "Physician Assistant"],
            "Law": ["Lawyer", "Judge", "Paralegal", "Legal Consultant", "Corporate Counsel", "Public Defender"],
            "Business": ["Business Analyst", "Entrepreneur", "Consultant", "Marketing Manager", "Financial Analyst", "Operations Manager"],
            "Education": ["Teacher", "School Administrator", "Educational Consultant", "Curriculum Developer", "Special Education Teacher", "Education Policy Analyst"],
            "Communications": ["Public Relations Specialist", "Communications Manager", "Journalist", "Media Planner", "Marketing Coordinator", "Content Strategist"],
            "Languages": ["Translator", "Interpreter", "Language Teacher", "Linguist", "Foreign Service Officer", "Localization Specialist"],
            "Theater": ["Actor", "Director", "Producer", "Playwright", "Drama Teacher", "Stage Manager"],
            "Dance": ["Dancer", "Choreographer", "Dance Teacher", "Dance Therapist", "Dance Company Director", "Dance Critic"],
            "Agricultural Sciences": ["Agricultural Scientist", "Farm Manager", "Agronomist", "Soil Scientist", "Food Scientist", "Extension Agent"],
            "Urban Planning": ["Urban Planner", "City Planner", "Transportation Planner", "Zoning Inspector", "Community Development Specialist", "Urban Designer"],
            "Nutrition": ["Dietitian", "Nutritionist", "Food Scientist", "Public Health Specialist", "Clinical Dietitian", "Health Educator"],
            "Physical Education": ["PE Teacher", "Coach", "Athletic Trainer", "Sports Administrator", "Fitness Instructor", "Recreation Director"],
            "Kinesiology": ["Exercise Physiologist", "Kinesiologist", "Rehabilitation Specialist", "Sports Scientist", "Fitness Consultant", "Occupational Therapist"],
            "Landscape Architecture": ["Landscape Architect", "Urban Planner", "Environmental Designer", "Garden Designer", "Site Planner", "Landscape Contractor"],
            "Jewish Studies": ["Rabbi", "Jewish Educator", "Historian", "Cultural Advisor", "Museum Curator", "Community Outreach Coordinator"],
            "Media and Cinema Studies": ["Film Critic", "Film Director", "Screenwriter", "Producer", "Media Planner", "Broadcast Technician"],
            "Media": ["Media Planner", "Broadcast Technician", "Content Creator", "Social Media Manager", "Journalist", "Public Relations Specialist"],
            "Aerospace Engineering": ["Aerospace Engineer", "Flight Engineer", "Aircraft Designer", "Propulsion Engineer", "Aviation Safety Specialist", "Satellite Engineer"],
            "African American Studies": ["Cultural Historian", "Civil Rights Advocate", "Community Organizer", "Professor", "Researcher", "Policy Analyst"],
            "African Studies": ["Diplomat", "Cultural Advisor", "International Development Specialist", "Researcher", "Professor", "Policy Analyst"],
            "Agricultural Communications": ["Agricultural Journalist", "Public Relations Specialist", "Marketing Coordinator", "Extension Agent", "Communications Director", "Content Strategist"],
            "Agricultural Education": ["Agricultural Educator", "Extension Agent", "Vocational Teacher", "Curriculum Developer", "Education Consultant", "Farm Manager"],
            "Agricultural Health and Safety": ["Safety Manager", "Agricultural Inspector", "Environmental Health Specialist", "Farm Safety Advisor", "Occupational Health Specialist"],
            "Agricultural Science Education": ["Agricultural Science Teacher", "Extension Agent", "Curriculum Developer", "Education Consultant", "Research Scientist"],
            "Animal Sciences": ["Veterinarian", "Animal Scientist", "Zoologist", "Animal Nutritionist", "Wildlife Biologist", "Animal Breeder"],
            "Arabic": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Foreign Service Officer"],
            "Archaeology": ["Archaeologist", "Museum Curator", "Cultural Resource Manager", "Professor", "Conservationist", "Research Scientist"],
            "Asian Studies": ["Diplomat", "Cultural Advisor", "International Business Specialist", "Researcher", "Professor", "Policy Analyst"],
            "Atmospheric Sciences": ["Meteorologist", "Climate Scientist", "Environmental Consultant", "Weather Forecaster", "Research Scientist", "Atmospheric Researcher"],
            "Basque": ["Translator", "Interpreter", "Cultural Advisor", "Language Teacher", "Researcher", "Professor"],
            "Behavioral Sciences": ["Behavioral Scientist", "Researcher", "Psychologist", "Human Resources Specialist", "Policy Analyst", "Market Research Analyst"],
            "Biochemistry": ["Biochemist", "Research Scientist", "Pharmacologist", "Biotechnologist", "Medical Scientist", "Clinical Researcher"],
            "Biophysics": ["Biophysicist", "Research Scientist", "Medical Scientist", "Biotechnologist", "Pharmaceutical Researcher", "Professor"],
            "Biostatistics": ["Biostatistician", "Data Analyst", "Public Health Researcher", "Clinical Trial Manager", "Epidemiologist", "Research Scientist"],
            "Biomedical Engineering": ["Biomedical Engineer", "Clinical Engineer", "Medical Device Designer", "Research Scientist", "Biotechnologist", "Rehabilitation Engineer"],
            "Business Administration": ["Business Manager", "Operations Manager", "Entrepreneur", "Consultant", "Marketing Manager", "Project Manager"],
            "Comparative World Literature": ["Literary Critic", "Translator", "Editor", "Professor", "Cultural Advisor", "Publisher"],
            "Classical Civilizations": ["Historian", "Archaeologist", "Museum Curator", "Professor", "Researcher", "Cultural Resource Manager"],
            "Dance Theater": ["Dancer", "Choreographer", "Dance Instructor", "Performance Artist", "Dance Company Director", "Dance Therapist"],
            "East Asian Languages and Cultures": ["Translator", "Interpreter", "Diplomat", "Language Teacher", "Cultural Advisor", "International Relations Specialist"],
            "Electrical and Computer Engineering": ["Electrical Engineer", "Computer Engineer", "Software Developer", "Systems Engineer", "Network Engineer", "Robotics Engineer"],
            "English": ["Writer", "Editor", "Teacher", "Journalist", "Publisher", "Literary Critic"],
            "Environmental Law": ["Environmental Lawyer", "Policy Analyst", "Environmental Consultant", "Public Advocate", "Compliance Officer", "Regulatory Affairs Specialist"],
            "Environmental Policy and Planning": ["Environmental Planner", "Urban Planner", "Policy Analyst", "Environmental Consultant", "Sustainability Coordinator", "Natural Resources Manager"],
            "Environmental Studies": ["Environmental Scientist", "Ecologist", "Conservationist", "Environmental Educator", "Wildlife Biologist", "Sustainability Coordinator"],
            "Family Health": ["Family Health Specialist", "Public Health Educator", "Nutritionist", "Family Therapist", "Health Educator", "Community Health Worker"],
            "Family Studies": ["Family Therapist", "Social Worker", "Human Development Specialist", "Child and Family Advocate", "Counselor", "Public Health Educator"],
            "French": ["Translator", "Interpreter", "Language Teacher", "Diplomat", "Cultural Advisor", "Tour Guide"],
            "Food Science and Human Nutrition": ["Dietitian", "Nutritionist", "Food Scientist", "Public Health Specialist", "Quality Control Specialist", "Product Developer"],
            "Geographic Information Systems": ["GIS Specialist", "Urban Planner", "Environmental Scientist", "Cartographer", "Remote Sensing Analyst", "Geospatial Analyst"],
            "German": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Global Studies": ["International Relations Specialist", "Policy Analyst", "Diplomat", "Researcher", "Humanitarian Aid Worker", "Global Health Specialist"],
            "Greek": ["Translator", "Interpreter", "Language Teacher", "Classical Studies Scholar", "Cultural Advisor", "Tour Guide"],
            "Gender and Women's Studies": ["Gender Studies Specialist", "Policy Analyst", "Advocate", "Social Worker", "Researcher", "Professor"],
            "Health and Human Services": ["Public Health Administrator", "Health Educator", "Social Worker", "Community Health Worker", "Public Health Analyst", "Health Policy Analyst"],
            "Hebrew": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Tour Guide"],
            "Hindi": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Tour Guide"],
            "Horticulture": ["Horticulturist", "Landscape Designer", "Botanist", "Environmental Consultant", "Greenhouse Manager", "Urban Farmer"],
            "Human Development and Family Studies": ["Family Therapist", "Social Worker", "Human Development Specialist", "Child and Family Advocate", "Counselor", "Public Health Educator"],
            "Integrative Biology": ["Biologist", "Research Scientist", "Environmental Scientist", "Ecologist", "Wildlife Biologist", "Conservationist"],
            "Italian": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Japanese": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Tour Guide"],
            "Korean": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Tour Guide"],
            "Latin": ["Translator", "Interpreter", "Classical Studies Scholar", "Language Teacher", "Cultural Advisor", "Museum Curator"],
            "Latin American and Caribbean Studies": ["Diplomat", "Cultural Advisor", "International Development Specialist", "Researcher", "Professor", "Policy Analyst"],
            "Labor and Employment Relations": ["Labor Relations Specialist", "Human Resources Manager", "Policy Analyst", "Union Representative", "Consultant", "Mediator"],
            "Leadership Studies": ["Leadership Coach", "Management Consultant", "Executive Director", "Training and Development Manager", "Organizational Development Specialist"],
            "Linguistics": ["Linguist", "Translator", "Interpreter", "Language Teacher", "Speech Pathologist", "Cultural Advisor"],
            "Latina/Latino Studies": ["Social Worker", "Community Organizer", "Cultural Advisor", "Policy Analyst", "Researcher", "Professor"],
            "Management": ["Business Manager", "Operations Manager", "Project Manager", "Consultant", "Entrepreneur", "Executive Director"],
            "Marketing": ["Marketing Manager", "Brand Manager", "Market Research Analyst", "Public Relations Specialist", "Advertising Executive", "Digital Marketing Specialist"],
            "Microbiology": ["Microbiologist", "Research Scientist", "Clinical Laboratory Scientist", "Biotechnologist", "Infection Control Specialist"],
            "Molecular and Cellular Biology": ["Molecular Biologist", "Research Scientist", "Biotechnologist", "Clinical Researcher", "Medical Scientist"],
            "Medieval Studies": ["Historian", "Archaeologist", "Museum Curator", "Professor", "Researcher", "Cultural Resource Manager"],
            "Nuclear Engineering": ["Nuclear Engineer", "Radiation Protection Specialist", "Reactor Operator", "Medical Physicist", "Research Scientist"],
            "Near Eastern Languages and Cultures": ["Translator", "Interpreter", "Diplomat", "Language Teacher", "Cultural Advisor", "International Relations Specialist"],
            "Neuroscience": ["Neuroscientist", "Research Scientist", "Clinical Neuropsychologist", "Pharmaceutical Researcher", "Cognitive Scientist"],
            "Natural Resources and Environmental Sciences": ["Environmental Scientist", "Conservationist", "Natural Resource Manager", "Wildlife Biologist", "Environmental Consultant"],
            "Pathobiology": ["Pathologist", "Research Scientist", "Clinical Laboratory Scientist", "Medical Researcher", "Infectious Disease Specialist"],
            "Persian": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Diplomat", "Tour Guide"],
            "Portuguese": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Quechua": ["Translator", "Interpreter", "Cultural Advisor", "Language Teacher", "Researcher", "Tour Guide"],
            "Rehabilitation Education": ["Rehabilitation Counselor", "Occupational Therapist", "Physical Therapist", "Special Education Teacher", "Vocational Rehabilitation Specialist"],
            "Religious Studies": ["Religious Educator", "Clergy Member", "Chaplain", "Researcher", "Professor", "Cultural Advisor"],
            "Recreation, Sport, and Tourism": ["Recreation Director", "Sports Manager", "Tourism Consultant", "Event Planner", "Parks and Recreation Manager"],
            "Russian": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Scandinavian": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Spanish": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide", "International Business Specialist"],
            "Special Education": ["Special Education Teacher", "Rehabilitation Counselor", "Educational Diagnostician", "School Counselor", "Advocate"],
            "Statistics": ["Statistician", "Data Scientist", "Biostatistician", "Actuary", "Market Research Analyst"],
            "Swahili": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide"],
            "Turkish": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide"],
            "Ukrainian": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide"],
            "Veterinary Medicine": ["Veterinarian", "Veterinary Technician", "Animal Scientist", "Wildlife Veterinarian", "Research Scientist"],
            "Women's Studies": ["Gender Studies Specialist", "Policy Analyst", "Advocate", "Social Worker", "Researcher", "Professor"],
            "World Literatures": ["Literary Critic", "Translator", "Editor", "Professor", "Cultural Advisor", "Publisher"],
            "Yiddish": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Researcher"],
            "Zulu": ["Translator", "Interpreter", "Language Teacher", "Cultural Advisor", "Tour Guide"]
        },
        'careers': [
            "Physicist", "Research Scientist", "Data Analyst", "Engineer", "Professor", "Astrophysicist",
            "Mathematician", "Statistician", "Actuary", "Data Scientist", "Operations Research Analyst", "Financial Analyst",
            "Biologist", "Microbiologist", "Geneticist", "Biotechnologist", "Environmental Scientist",
            "Chemist", "Pharmacologist", "Forensic Scientist", "Chemical Engineer", "Toxicologist", "Materials Scientist",
            "Historian", "Archivist", "Museum Curator", "Teacher", "Researcher", "Historical Consultant",
            "Writer", "Editor", "Publisher", "Journalist", "Literary Critic",
            "Software Developer", "Cybersecurity Specialist", "Systems Analyst", "AI Engineer", "Database Administrator",
            "Artist", "Graphic Designer", "Art Director", "Illustrator",
            "Musician", "Music Teacher", "Composer", "Sound Engineer", "Music Therapist", "Conductor",
            "Economist", "Financial Analyst", "Policy Analyst", "Consultant", "Market Research Analyst",
            "Psychologist", "Therapist", "Counselor", "Human Resources Specialist", "Clinical Psychologist",
            "Sociologist", "Social Worker", "Urban Planner", "Community Service Manager",
            "Anthropologist", "Archaeologist", "Cultural Resource Manager", "Forensic Anthropologist",
            "Politician", "Public Relations Specialist", "Diplomat", "Political Consultant", "Lobbyist",
            "Philosopher", "Ethicist",
            "Ecologist", "Conservationist", "Environmental Consultant", "Wildlife Biologist", "Environmental Engineer",
            "Geologist", "Hydrologist", "Mining Engineer", "Seismologist", "Petroleum Geologist",
            "Astronomer", "Planetarium Director",
            "Civil Engineer", "Mechanical Engineer", "Electrical Engineer", "Software Engineer", "Aerospace Engineer", "Biomedical Engineer",
            "Doctor", "Surgeon", "Nurse", "Medical Researcher", "Pharmacist", "Physician Assistant",
            "Lawyer", "Judge", "Paralegal", "Legal Consultant", "Corporate Counsel", "Public Defender",
            "Business Analyst", "Entrepreneur", "Marketing Manager", "Operations Manager",
            "School Administrator", "Educational Consultant", "Special Education Teacher", "Education Policy Analyst",
            "Communications Manager", "Media Planner", "Marketing Coordinator", "Content Strategist",
            "Translator", "Interpreter", "Language Teacher", "Foreign Service Officer", "Localization Specialist",
            "Actor", "Director", "Producer", "Playwright", "Drama Teacher", "Stage Manager",
            "Dancer", "Choreographer", "Dance Teacher", "Dance Therapist", "Dance Company Director", "Dance Critic",
            "Agricultural Scientist", "Farm Manager", "Agronomist", "Soil Scientist", "Food Scientist", "Extension Agent",
            "City Planner", "Transportation Planner", "Zoning Inspector", "Community Development Specialist", "Urban Designer",
            "Dietitian", "Nutritionist", "Public Health Specialist", "Clinical Dietitian", "Health Educator",
            "PE Teacher", "Coach", "Athletic Trainer", "Sports Administrator", "Fitness Instructor", "Recreation Director",
            "Exercise Physiologist", "Kinesiologist", "Rehabilitation Specialist", "Sports Scientist", "Fitness Consultant", "Occupational Therapist",
            "Landscape Architect", "Environmental Designer", "Garden Designer", "Site Planner", "Landscape Contractor",
            "Rabbi", "Jewish Educator", "Cultural Advisor", "Community Outreach Coordinator",
            "Film Critic", "Film Director", "Screenwriter", "Producer", "Broadcast Technician",
            "Content Creator", "Social Media Manager",
            "Flight Engineer", "Aircraft Designer", "Propulsion Engineer", "Satellite Engineer",
            "Cultural Historian", "Civil Rights Advocate", "Community Organizer",
            "International Business Specialist",
            "Agricultural Journalist", "Marketing Coordinator", "Communications Director",
            "Agricultural Educator", "Vocational Teacher",
            "Safety Manager", "Agricultural Inspector", "Occupational Health Specialist",
            "Animal Scientist", "Animal Nutritionist", "Animal Breeder",
            "Tour Guide",
            "Meteorologist", "Climate Scientist", "Weather Forecaster", "Atmospheric Researcher",
            "Behavioral Scientist",
            "Biochemist", "Medical Scientist", "Clinical Researcher",
            "Biophysicist",
            "Public Health Researcher", "Clinical Trial Manager", "Epidemiologist",
            "Clinical Engineer", "Medical Device Designer", "Rehabilitation Engineer",
            "Project Manager",
            "Dancer", "Dance Instructor", "Performance Artist",
            "International Relations Specialist",
            "Systems Engineer", "Network Engineer", "Robotics Engineer",
            "Public Advocate", "Compliance Officer", "Regulatory Affairs Specialist",
            "Environmental Educator", "Sustainability Coordinator",
            "Family Health Specialist", "Family Therapist", "Community Health Worker",
            "Child and Family Advocate",
            "Quality Control Specialist", "Product Developer",
            "GIS Specialist", "Cartographer", "Remote Sensing Analyst", "Geospatial Analyst",
            "Humanitarian Aid Worker", "Global Health Specialist",
            "Classical Studies Scholar",
            "Gender Studies Specialist",
            "Public Health Administrator", "Health Policy Analyst",
            "Horticulturist", "Landscape Designer", "Botanist", "Greenhouse Manager", "Urban Farmer",
            "Human Development Specialist",
            "Natural Resource Manager",
            "Pathologist", "Clinical Laboratory Scientist", "Infectious Disease Specialist",
            "Rehabilitation Counselor", "Vocational Rehabilitation Specialist",
            "Religious Educator", "Clergy Member", "Chaplain",
            "Recreation Director", "Sports Manager", "Tourism Consultant", "Event Planner", "Parks and Recreation Manager",
            "School Counselor", "Educational Diagnostician",
            "Veterinary Technician", "Wildlife Veterinarian",
            "Executive Director",
            "Leadership Coach", "Management Consultant", "Training and Development Manager", "Organizational Development Specialist",
            "Social Worker", "Community Organizer"
        ],
        'related_topics': {
            "Physics": ["Biophysics", "Astrophysics", "Quantum Mechanics", "Relativity", "Thermodynamics", "Engineering"],
            "Mathematics": ["Statistics", "Biostatistics", "Applied Mathematics", "Algebra", "Calculus", "Computer Science"],
            "Biology": ["Biochemistry", "Molecular and Cellular Biology", "Integrative Biology", "Microbiology", "Neuroscience", "Environmental Science"],
            "Chemistry": ["Biochemistry", "Chemical Engineering", "Pharmacology", "Environmental Science", "Material Science"],
            "History": ["Political Science", "Archaeology", "Classical Civilizations", "Anthropology", "Medieval Studies"],
            "Literature": ["Comparative World Literature", "English", "Classical Civilizations", "World Literatures", "Cultural Studies"],
            "Computer Science": ["Software Engineering", "Data Science", "Artificial Intelligence", "Cybersecurity", "Electrical and Computer Engineering"],
            "Art": ["Art History", "Graphic Design", "Film and Media Studies", "Theater", "Dance"],
            "Music": ["Music Theory", "Performance Studies", "Ethnomusicology", "Sound Engineering", "Dance Theater"],
            "Economics": ["Business", "Finance", "Political Science", "Statistics", "Behavioral Sciences"],
            "Psychology": ["Behavioral Sciences", "Neuroscience", "Cognitive Science", "Sociology", "Education"],
            "Sociology": ["Anthropology", "Psychology", "Political Science", "Gender and Women's Studies", "Cultural Studies"],
            "Anthropology": ["Sociology", "Archaeology", "Cultural Studies", "Linguistics", "History"],
            "Political Science": ["History", "Law", "Economics", "International Relations", "Public Policy"],
            "Philosophy": ["Ethics", "Logic", "Political Science", "History", "Religious Studies"],
            "Environmental Science": ["Ecology", "Geology", "Atmospheric Sciences", "Agricultural Sciences", "Natural Resources and Environmental Sciences"],
            "Geology": ["Earth Sciences", "Environmental Science", "Geography", "Natural Resources and Environmental Sciences", "Atmospheric Sciences"],
            "Astronomy": ["Astrophysics", "Physics", "Mathematics", "Space Science", "Engineering"],
            "Engineering": ["Mechanical Engineering", "Electrical and Computer Engineering", "Civil Engineering", "Biomedical Engineering", "Aerospace Engineering"],
            "Medicine": ["Biomedical Engineering", "Biology", "Pharmacology", "Neuroscience", "Health and Human Services"],
            "Law": ["Political Science", "Economics", "Philosophy", "Public Policy", "Environmental Law"],
            "Business": ["Economics", "Management", "Marketing", "Finance", "Business Administration"],
            "Education": ["Psychology", "Sociology", "Special Education", "Leadership Studies", "Human Development and Family Studies"],
            "Communications": ["Journalism", "Media and Cinema Studies", "Marketing", "Political Science", "Linguistics"],
            "Languages": ["Linguistics", "Literature", "Cultural Studies", "Anthropology", "Education"],
            "Theater": ["Drama", "Dance Theater", "Film and Media Studies", "Literature", "Cultural Studies"],
            "Dance": ["Theater", "Physical Education", "Music", "Dance Theater", "Performance Studies"],
            "Agricultural Sciences": ["Environmental Science", "Animal Sciences", "Food Science and Human Nutrition", "Agricultural Education", "Natural Resources and Environmental Sciences"],
            "Urban Planning": ["Architecture", "Civil Engineering", "Environmental Policy and Planning", "Public Policy", "Geography"],
            "Nutrition": ["Food Science and Human Nutrition", "Biochemistry", "Public Health", "Medicine", "Kinesiology"],
            "Physical Education": ["Kinesiology", "Health and Fitness", "Sports Science", "Recreation, Sport, and Tourism", "Dance"],
            "Kinesiology": ["Physical Education", "Sports Science", "Medicine", "Rehabilitation Education", "Health and Human Services"],
            "Landscape Architecture": ["Urban Planning", "Architecture", "Environmental Design", "Horticulture", "Geography"],
            "Jewish Studies": ["Religious Studies", "History", "Cultural Studies", "Hebrew", "Anthropology"],
            "Media and Cinema Studies": ["Film Studies", "Communications", "Journalism", "Cultural Studies", "Art"],
            "Media": ["Communications", "Journalism", "Marketing", "Digital Media", "Graphic Design"],
            "Aerospace Engineering": ["Mechanical Engineering", "Physics", "Electrical Engineering", "Mathematics", "Computer Science"],
            "African American Studies": ["History", "Sociology", "Cultural Studies", "Political Science", "Literature"],
            "African Studies": ["Anthropology", "History", "Political Science", "Cultural Studies", "Sociology"],
            "Agricultural Communications": ["Agricultural Sciences", "Communications", "Journalism", "Public Relations", "Environmental Policy and Planning"],
            "Agricultural Education": ["Agricultural Sciences", "Education", "Environmental Science", "Human Development and Family Studies", "Leadership Studies"],
            "Agricultural Health and Safety": ["Agricultural Sciences", "Environmental Health", "Public Health", "Medicine", "Safety Engineering"],
            "Agricultural Science Education": ["Agricultural Education", "Agricultural Sciences", "Education", "Environmental Science", "Leadership Studies"],
            "Animal Sciences": ["Biology", "Veterinary Medicine", "Agricultural Sciences", "Environmental Science", "Animal Behavior"],
            "Arabic": ["Languages", "Cultural Studies", "Middle Eastern Studies", "Linguistics", "Literature"],
            "Archaeology": ["Anthropology", "History", "Classical Civilizations", "Geography", "Cultural Studies"],
            "Asian Studies": ["Languages", "Cultural Studies", "History", "Political Science", "Literature"],
            "Atmospheric Sciences": ["Environmental Science", "Geology", "Geography", "Climate Science", "Physics"],
            "Basque": ["Languages", "Cultural Studies", "Anthropology", "History", "Literature"],
            "Behavioral Sciences": ["Psychology", "Sociology", "Cognitive Science", "Neuroscience", "Education"],
            "Biochemistry": ["Chemistry", "Biology", "Molecular and Cellular Biology", "Medicine", "Pharmacology"],
            "Biophysics": ["Physics", "Biology", "Biochemistry", "Molecular and Cellular Biology", "Medicine"],
            "Biostatistics": ["Statistics", "Mathematics", "Public Health", "Epidemiology", "Data Science"],
            "Biomedical Engineering": ["Engineering", "Biology", "Medicine", "Biophysics", "Neuroscience"],
            "Business Administration": ["Business", "Management", "Economics", "Marketing", "Finance"],
            "Comparative World Literature": ["Literature", "Cultural Studies", "Languages", "History", "Anthropology"],
            "Classical Civilizations": ["History", "Archaeology", "Anthropology", "Literature", "Languages"],
            "Dance Theater": ["Dance", "Theater", "Music", "Performance Studies", "Cultural Studies"],
            "East Asian Languages and Cultures": ["Asian Studies", "Languages", "History", "Cultural Studies", "Literature"],
            "Electrical and Computer Engineering": ["Engineering", "Computer Science", "Physics", "Mathematics", "Robotics"],
            "English": ["Literature", "Cultural Studies", "Languages", "History", "Education"],
            "Environmental Law": ["Law", "Environmental Policy and Planning", "Environmental Studies", "Public Policy", "Political Science"],
            "Environmental Policy and Planning": ["Urban Planning", "Environmental Studies", "Public Policy", "Geography", "Environmental Science"],
            "Environmental Studies": ["Environmental Science", "Geology", "Geography", "Natural Resources and Environmental Sciences", "Environmental Policy and Planning"],
            "Family Health": ["Health and Human Services", "Nutrition", "Public Health", "Medicine", "Family Studies"],
            "Family Studies": ["Human Development and Family Studies", "Sociology", "Psychology", "Education", "Family Health"],
            "French": ["Languages", "Literature", "Cultural Studies", "History", "Comparative World Literature"],
            "Food Science and Human Nutrition": ["Nutrition", "Biochemistry", "Public Health", "Agricultural Sciences", "Medicine"],
            "Geographic Information Systems": ["Geography", "Environmental Science", "Urban Planning", "Geology", "Data Science"],
            "German": ["Languages", "Literature", "Cultural Studies", "History", "Comparative World Literature"],
            "Global Studies": ["International Relations", "Political Science", "Economics", "Cultural Studies", "History"],
            "Greek": ["Languages", "Classical Civilizations", "Literature", "History", "Cultural Studies"],
            "Gender and Women's Studies": ["Sociology", "Anthropology", "History", "Political Science", "Cultural Studies"],
            "Health and Human Services": ["Public Health", "Medicine", "Family Studies", "Social Work", "Nutrition"],
            "Hebrew": ["Languages", "Jewish Studies", "Religious Studies", "Literature", "Cultural Studies"],
            "Hindi": ["Languages", "Cultural Studies", "Literature", "Asian Studies", "Linguistics"],
            "Horticulture": ["Agricultural Sciences", "Environmental Science", "Botany", "Landscape Architecture", "Urban Planning"],
            "Human Development and Family Studies": ["Family Studies", "Education", "Psychology", "Sociology", "Social Work"],
            "Integrative Biology": ["Biology", "Biochemistry", "Molecular and Cellular Biology", "Neuroscience", "Environmental Science"],
            "Italian": ["Languages", "Literature", "Cultural Studies", "History", "Comparative World Literature"],
            "Japanese": ["Languages", "Cultural Studies", "Literature", "Asian Studies", "Linguistics"],
            "Korean": ["Languages", "Cultural Studies", "Literature", "Asian Studies", "Linguistics"],
            "Latin": ["Languages", "Classical Civilizations", "Literature", "History", "Cultural Studies"],
            "Latin American and Caribbean Studies": ["Anthropology", "Political Science", "Cultural Studies", "History", "Sociology"],
            "Labor and Employment Relations": ["Sociology", "Economics", "Political Science", "Management", "Law"],
            "Leadership Studies": ["Management", "Education", "Sociology", "Psychology", "Political Science"],
            "Linguistics": ["Languages", "Cognitive Science", "Anthropology", "Cultural Studies", "Communications"],
            "Latina/Latino Studies": ["Sociology", "Anthropology", "Cultural Studies", "Political Science", "History"],
            "Management": ["Business", "Economics", "Leadership Studies", "Marketing", "Business Administration"],
            "Marketing": ["Business", "Economics", "Communications", "Management", "Business Administration"],
            "Microbiology": ["Biology", "Biochemistry", "Molecular and Cellular Biology", "Medicine", "Environmental Science"],
            "Molecular and Cellular Biology": ["Biology", "Biochemistry", "Microbiology", "Neuroscience", "Medicine"],
            "Medieval Studies": ["History", "Classical Civilizations", "Literature", "Archaeology", "Cultural Studies"],
            "Nuclear Engineering": ["Engineering", "Physics", "Mechanical Engineering", "Electrical and Computer Engineering", "Mathematics"],
            "Near Eastern Languages and Cultures": ["Middle Eastern Studies", "Languages", "History", "Cultural Studies", "Religious Studies"],
            "Neuroscience": ["Biology", "Psychology", "Medicine", "Cognitive Science", "Molecular and Cellular Biology"],
            "Natural Resources and Environmental Sciences": ["Environmental Science", "Geology", "Agricultural Sciences", "Geography", "Atmospheric Sciences"],
            "Pathobiology": ["Biology", "Medicine", "Microbiology", "Public Health", "Biochemistry"],
            "Persian": ["Languages", "Middle Eastern Studies", "Literature", "Cultural Studies", "History"],
            "Portuguese": ["Languages", "Literature", "Cultural Studies", "Latin American Studies", "Comparative World Literature"],
            "Quechua": ["Languages", "Latin American Studies", "Anthropology", "History", "Cultural Studies"],
            "Rehabilitation Education": ["Education", "Physical Education", "Kinesiology", "Health and Human Services", "Special Education"],
            "Religious Studies": ["Philosophy", "History", "Anthropology", "Cultural Studies", "Sociology"],
            "Recreation, Sport, and Tourism": ["Physical Education", "Kinesiology", "Business", "Health and Human Services", "Economics"],
            "Russian": ["Languages", "Literature", "Cultural Studies", "History", "Comparative World Literature"],
            "Scandinavian": ["Languages", "Literature", "Cultural Studies", "History", "Comparative World Literature"],
            "Spanish": ["Languages", "Literature", "Cultural Studies", "Latin American Studies", "Comparative World Literature"],
            "Special Education": ["Education", "Psychology", "Rehabilitation Education", "Sociology", "Human Development and Family Studies"],
            "Statistics": ["Mathematics", "Biostatistics", "Data Science", "Economics", "Computer Science"],
            "Swahili": ["Languages", "Cultural Studies", "African Studies", "Literature", "History"],
            "Turkish": ["Languages", "Middle Eastern Studies", "Literature", "Cultural Studies", "History"],
            "Ukrainian": ["Languages", "Cultural Studies", "History", "Comparative World Literature", "Literature"],
            "Veterinary Medicine": ["Animal Sciences", "Biology", "Medicine", "Pathobiology", "Environmental Science"],
            "Women's Studies": ["Gender Studies", "Sociology", "Anthropology", "History", "Political Science"],
            "World Literatures": ["Literature", "Comparative World Literature", "Languages", "Cultural Studies", "History"],
            "Yiddish": ["Languages", "Jewish Studies", "Literature", "Cultural Studies", "History"],
            "Zulu": ["Languages", "Cultural Studies", "African Studies", "Literature", "History"]
        },
        'course_type_to_learning_styles': {
            'Discussion/Recitation': ['Auditory', 'Read/Write'],
            'Lecture': ['Visual', 'Auditory', 'Read/Write'],
            'Lecture-Discussion': ['Visual', 'Auditory', 'Read/Write'],
            'Online': ['Visual', 'Read/Write'],
            'Independent Study': ['Read/Write'],
            'Laboratory': ['Kinesthetic'],
            'Online Lecture': ['Visual', 'Auditory'],
            'Online Discussion': ['Visual', 'Auditory'],
            'Online Lecture Discussion': ['Visual', 'Auditory'],
            'Conference': ['Auditory'],
            'Laboratory-Discussion': ['Kinesthetic', 'Read/Write'],
            'Online Lab': ['Visual', 'Kinesthetic'],
            'Study Abroad': ['Visual', 'Auditory', 'Kinesthetic'],
            'Internship': ['Visual', 'Auditory', 'Kinesthetic'],
            'Seminar': ['Auditory', 'Read/Write'],
            'Travel': ['Visual', 'Auditory', 'Kinesthetic'],
            'Research': ['Kinesthetic', 'Read/Write'],
            'Studio': ['Visual', 'Auditory'],
            'Packaged Section': ['Visual', 'Auditory', 'Read/Write'],
            'Quiz': ['Read/Write'],
            'Practice': ['Visual', 'Auditory', 'Kinesthetic']
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
            'Female': 54.83,
            'Male': 40.07,
            'Nonbinary': 5.1
        },
        'international': {
            'Domestic': 94.08,
            'International': 5.92
        },
        'socioeconomic': {
            'In poverty': 20,
            'Near poverty': 19,
            'Lower-middle income': 15,
            'Middle income': 37,
            'Higher income': 9
        },
        'learning_style': {
            'Visual': 27.27,
            'Auditory': 23.56,
            'Read/Write': 21.16,
            'Kinesthetic': 28.01
        },
        'future_topics': [
            "Biophysics", "Astrophysics", "Quantum Mechanics", "Relativity", "Thermodynamics", "Engineering",
            "Statistics", "Biostatistics", "Applied Mathematics", "Algebra", "Calculus", "Computer Science",
            "Biochemistry", "Molecular and Cellular Biology", "Integrative Biology", "Microbiology", "Neuroscience", "Environmental Science",
            "Chemical Engineering", "Pharmacology", "Material Science",
            "Political Science", "Archaeology", "Classical Civilizations", "Anthropology", "Medieval Studies",
            "Comparative World Literature", "English", "Cultural Studies",
            "Software Engineering", "Data Science", "Artificial Intelligence", "Cybersecurity", "Electrical and Computer Engineering",
            "Art History", "Graphic Design", "Film and Media Studies", "Theater", "Dance",
            "Music Theory", "Performance Studies", "Ethnomusicology", "Sound Engineering",
            "Business", "Finance", "Policy Analysis", "Behavioral Sciences",
            "Cognitive Science", "Human Resources", "Clinical Psychology",
            "Urban Planning", "Community Service Management",
            "Cultural Resource Management", "Forensic Anthropology",
            "International Relations", "Public Policy",
            "Ethics", "Logic",
            "Ecology", "Geology", "Atmospheric Sciences", "Natural Resources and Environmental Sciences",
            "Earth Sciences", "Geography", "Climate Science",
            "Space Science",
            "Mechanical Engineering", "Civil Engineering", "Biomedical Engineering", "Aerospace Engineering",
            "Public Health",
            "Compliance", "Corporate Law",
            "Marketing", "Operations Management",
            "Special Education", "Curriculum Development",
            "Journalism", "Media Planning", "Content Strategy",
            "Translation", "Interpretation", "Localization",
            "Drama", "Performance Arts",
            "Sports Science", "Recreation",
            "Landscape Design", "Urban Agriculture",
            "Rabbinical Studies", "Jewish Education",
            "Film Studies", "Screenwriting", "Broadcast Technology",
            "Digital Media",
            "Flight Engineering", "Aviation Safety",
            "Civil Rights Advocacy", "Community Outreach",
            "International Development",
            "Public Relations",
            "Agronomy", "Soil Science", "Extension",
            "Safety Engineering",
            "Animal Behavior",
            "Middle Eastern Studies",
            "Botany",
            "Wildlife Biology",
            "Pharmaceutical Research",
            "Clinical Trials",
            "Rehabilitation Engineering",
            "Project Management",
            "Linguistics", "Paleontology",
            "Parks and Recreation Management"
        ],
        'parameters': {
            'Random': ['Random', self.noise_level],
            'Gaussian': ['Gaussian', self.epsilon, self.delta, self.sensitivity],
            'Laplace': ['Laplace', self.epsilon, self.sensitivity],
            'Exponential': ['Exponential', self.epsilon, self.scale],
            'Gamma': ['Gamma', self.shape, self.scale],
            'Uniform': ['Uniform', self.low, self.high],
            'CBA': ['CBA', self.noise_level],
            'DDP': ['DDP', self.epsilon],
            'Pufferfish': ['Pufferfish', self.noise_level],
            'Poisson': ['Poisson', self.noise_level],
            'SaltAndPepper': ['SaltAndPepper', self.salt_prob, self.pepper_prob],
            'Speckle': ['Speckle', self.variance],
            'BitFlip': ['BitFlip', self.flip_prob],
            'AWGN': ['AWGN', self.snr],
            'Multiplicative': ['Multiplicative', self.variance],
        }
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
                    'European American or white': 'Settler Colonizer',
                    'Latino/a/x American': 'Settler Colonizer',
                    'African American or Black': 'Settler Colonizer',
                    'Asian American': 'Settler Colonizer',
                    'Multiracial': 'Settler Colonizer',
                    'American Indian or Alaska Native': 'Indigenous',
                    'Pacific Islander': 'Indigenous'
                },
                'slight': {
                    'European American or white': 'Settler Colonizer Nonminority',
                    'Latino/a/x American': 'Settler Colonizer Minority',
                    'African American or Black': 'Settler Colonizer Minority',
                    'Asian American': 'Settler Colonizer Minority',
                    'Multiracial': 'Settler Colonizer Minority',
                    'American Indian or Alaska Native': 'Indigenous',
                    'Pacific Islander': 'Indigenous'
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
                    'In poverty': 'Generalized',
                    'Near poverty': 'Generalized',
                    'Lower-middle income': 'Generalized',
                    'Middle income': 'Generalized',
                    'Higher income': 'Generalized'
                },
                'broad': {
                    'In poverty': 'Lower',
                    'Near poverty': 'Lower',
                    'Lower-middle income': 'Lower',
                    'Middle income': 'Upper',
                    'Higher income': 'Upper'
                },
                'slight': {
                    'In poverty': 'Lower',
                    'Near poverty': 'Lower',
                    'Lower-middle income': 'Middle',
                    'Middle income': 'Middle',
                    'Higher income': 'Upper'
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