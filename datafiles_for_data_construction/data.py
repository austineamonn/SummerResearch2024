import json

class Data:
    def __init__(self) -> None:
        pass

    def first_name(self):
        # Reading from a JSON file
        with open('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/datafiles_for_data_construction/first_names.json', 'r') as f:
            first_names = json.load(f)

        first_name = {
            'first_name': first_names
        }

        return first_name
    
    def last_name(self):
        # Reading from a JSON file
        with open('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/datafiles_for_data_construction/last_names.json', 'r') as f:
            last_names = json.load(f)

        last_name = {
            'lastt_name': last_names
        }

        return last_name

    def ethnoracial_group(self):
        race_ethnicity = {
            'ethnoracial_stats': {
                'European American or white': 53.4,
                'Latino/a/x American': 20.6,
                'African American or Black': 13.1,
                'Asian American': 7.6,
                'Multiracial': 4.3,
                'American Indian or Alaska Native': 0.7,
                'Pacific Islander': 0.3
            }
        }

        return race_ethnicity
    
    def gender(self):
        gender = {
            'gender': {
                'Female': 54.83,
                'Male': 40.07,
                'Nonbinary': 5.1
            }
        }

        return gender
    
    def international_status(self):
        status = {
            'international': {
                'Domestic': 94.08,
                'International': 5.92
            }
        }

        return status
    
    def socioeconomics_status(self):
        SES = {
            'socioeconomic': {
                'In poverty': 20,
                'Near poverty': 19,
                'Lower-middle income': 15,
                'Middle income': 37,
                'Higher income': 9
            }
        }

        return SES
    
    def learning_style(self):
        LS = {
            'learning_style': {
                'Visual': 27.27,
                'Auditory': 23.56,
                'Read/Write': 21.16,
                'Kinesthetic': 28.01
            }
        }

        return LS
    
    def major(self):
        major = {
            'majors_list': [
                'Accounting', 'Actuarial Science', 'Advertising And Public Relations',
                'Aerospace Engineering', 'Agricultural Economics', 'Agriculture Production And Management',
                'Animal Sciences', 'Anthropology And Archeology', 'Applied Mathematics',
                'Architectural Engineering', 'Architecture', 'Area Ethnic And Civilization Studies',
                'Art And Music Education', 'Art History And Criticism', 'Astronomy And Astrophysics',
                'Atmospheric Sciences And Meteorology', 'Biochemical Sciences', 'Biological Engineering',
                'Biology', 'Biomedical Engineering', 'Botany', 'Business Economics',
                'Business Management And Administration', 'Chemical Engineering', 'Chemistry',
                'Civil Engineering', 'Clinical Psychology', 'Cognitive Science And Biopsychology',
                'Commercial Art And Graphic Design', 'Communication Disorders Sciences And Services',
                'Communication Technologies', 'Communications', 'Community And Public Health',
                'Composition And Rhetoric', 'Computer Administration Management And Security',
                'Computer And Information Systems', 'Computer Engineering',
                'Computer Networking And Telecommunications', 'Computer Programming And Data Processing',
                'Computer Science', 'Construction Services', 'Cosmetology Services And Culinary Arts',
                'Counseling Psychology', 'Court Reporting', 'Criminal Justice And Fire Protection',
                'Criminology', 'Drama And Theater Arts', 'Early Childhood Education', 'Ecology',
                'Economics', 'Educational Administration And Supervision', 'Educational Psychology',
                'Electrical Engineering', 'Electrical Engineering Technology',
                'Electric, Mechanical, And Precision Tech', 'Elementary Education'
                'Engineering And Industrial Management', 'Engineering Mechanics Physics And Science',
                'Engineering Technologies', 'English Language And Literature',
                'Environmental Engineering', 'Environmental Science', 'Family And Consumer Sciences',
                'Film Video And Photographic Arts', 'Finance', 'Fine Arts', 'Food Science', 'Forestry',
                'French German And Other Language Studies', 'General Agriculture', 'General Business',
                'General Education', 'General Engineering', 'General Medical And Health Services',
                'General Social Sciences', 'Genetics', 'Geography', 'Geological And Geophysical Engineering',
                'Geology And Earth Science', 'Geosciences', 'Health And Medical Administrative Services',
                'Health And Medical Preparatory Programs', 'History', 'Hospitality Management',
                'Human Resources And Personnel Management', 'Human Services And Community Organization',
                'Humanities', 'Industrial And Manufacturing Engineering', 'Industrial And Organizational Psychology',
                'Industrial Production Technologies', 'Information Sciences',
                'Intercultural And International Studies', 'Interdisciplinary Social Sciences',
                'International Business', 'International Relations', 'Journalism',
                'Language And Drama Education', 'Liberal Arts', 'Library Science',
                'Linguistics And Comparative Language And Literature', 'Management Information Systems And Statistics',
                'Marketing And Marketing Research', 'Mass Media', 'Materials Engineering And Materials Science',
                'Materials Science', 'Mathematics', 'Mathematics And Computer Science', 'Mathematics Teacher Education',
                'Mechanical Engineering', 'Mechanical Engineering Related Technologies', 'Medical Assisting Services',
                'Medical Technologies Technicians', 'Metallurgical Engineering', 'Microbiology', 'Military Technologies',
                'Mining And Mineral Engineering', 'Miscellaneous Agriculture', 'Miscellaneous Biology',
                'Miscellaneous Business And Medical Administration', 'Miscellaneous Education',
                'Miscellaneous Engineering', 'Miscellaneous Engineering Technologies', 'Miscellaneous Fine Arts',
                'Miscellaneous Health Medical Professions', 'Miscellaneous Psychology', 'Miscellaneous Social Sciences',
                'Molecular Biology', 'Multi-Disciplinary Or General Science', 'Multi/Interdisciplinary Studies',
                'Music', 'Natural Resources Management', 'Naval Architecture And Marine Engineering', 'Neuroscience',
                'Nuclear Engineering', 'Nuclear, Industrial Radiology, And Biological Technologies', 'Nursing',
                'Nutrition Sciences', 'Oceanography', 'Operations Logistics And E-Commerce', 'Other Foreign Languages',
                'Petroleum Engineering', 'Pharmacology', 'Pharmacy Pharmaceutical Sciences And Administration',
                'Philosophy And Religious Studies', 'Physical And Health Education Teaching',
                'Physical Fitness Parks Recreation And Leisure', 'Physical Sciences', 'Physics', 'Physiology',
                'Plant Science And Agronomy', 'Political Science And Government', 'Pre-Law And Legal Studies',
                'Psychology', 'Public Administration', 'Public Policy', 'School Student Counseling',
                'Science And Computer Teacher Education', 'Secondary Teacher Education', 'Social Psychology',
                'Social Science Or History Teacher Education', 'Social Work', 'Sociology', 'Soil Science',
                'Special Needs Education', 'Statistics And Decision Science', 'Studio Arts',
                'Teacher Education: Multiple Levels', 'Theology And Religious Vocations',
                'Transportation Sciences And Technologies', 'Treatment Therapy Professions',
                'United States History', 'Visual And Performing Arts', 'Zoology'
            ]
        }

        return major
    
    def course(self):
        # Reading from a JSON file
        with open('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/datafiles_for_data_construction/courses.json', 'r') as f:
            course_tuples = json.load(f)

        courses = {
            'course_list': course_tuples
        }

        return courses
    
    def subjects(self):
        subjects =  {
            'subjects_list': [
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
            ]
        }

        return subjects
    
    def careers(self):
        careers = {
            'careers_list': [
                'Accountant and Auditor', 'Actor, Producer, and Director', 'Actuary', 'Aerospace Engineer',
                'Agricultural and Food Scientist', 'Agricultural worker,',
                'Air Traffic Controller and Airfield Operation Specialist', 'Aircraft Mechanic and Service Technician',
                'Aircraft Pilot and Flight Engineer', 'Architect, Except Naval', 'Artist and Related Worker',
                'Athlete, Coache, Umpire, and Related Worker', 'Atmospheric and Space Scientist',
                'Automotive Service Technician and Mechanic', 'Bartender', 'Bill and Account Collector',
                'Biological Scientist', 'Bookkeeping, Accounting, and Auditing Clerk',
                'Broadcast and Sound Engineering Technician and Radio Operator, and media and communication equipment worker',
                'Bus and Truck Mechanic and Diesel Engine Specialist', 'Cashier', 'Chef and Cook', 'Chemical Engineer',
                'Chemical Technician', 'Chemist and Material Scientist',
                'Chief executive and legislator/public administration', 'Childcare Worker', 'Civil Engineer',
                'Claim Adjuster, Appraiser, Examiner, and Investigator', 'Clergy',
                'Clinical Laboratory Technologist and Technician',
                'Combined Food Preparation and Serving Worker,Including Fast Food',
                'Computer Programmer', 'Computer Scientist and System Analyst/Network system Analyst/Web Developer',
                'Computer Support Specialist', 'Conservation Scientist and Forester', 'Construction Manager', 'Cost Estimator',
                'Counselor', 'Credit Counselor and Loan Officer', 'Customer Service Representative',
                'Dancer and Choreographer', 'Dental Hygienist', 'Designer', 'Diagnostic Related Technologist and Technician',
                'Dietician and Nutritionist', 'Director, Religious Activities and Education', 'Drafter',
                'Editor, New Analyst, Reporter, and Correspondent', 'Electrical and Electronic Engineer',
                'Elementary and Middle School Teacher', 'Engineer,', 'Environmental Engineer',
                'Environmental Scientist and Geoscientist', 'Farmer, Rancher, and Other Agricultural Manager',
                'Financial Analyst', 'Financial Manager', 'Firefighter', 'First-Line Enlisted Military Supervisor',
                'First-Line Supervisor of Construction Trade and Extraction Worker',
                'First-Line Supervisor of Food Preparation and Serving Worker',
                'First-Line Supervisor of Production and Operating Worker', 'First-Line Supervisor of Sale Worker',
                'Food Service and Lodging Manager', 'General and Operation Manager', 'Ground Maintenance Worker',
                'Health Diagnosing and Treating Practitioner Support Technician',
                'Healthcare Practitioner and Technical Occupation,', 'Human Resource Manager',
                'Human Resource, Training, and Labor Relation Specialist', 'Industrial Engineer, including Health and Safety',
                'Industrial and Refractory Machinery Mechanic', 'Inspector, Tester, Sorter, Sampler, and Weigher',
                'Insurance Sale Agent', 'Jeweler and Preciou Stone and Metal Worker',
                'Laborer and Freight, Stock, and Material Mover, Hand', 'Legal Support Worker,', 'Librarian',
                'Life, Physical, and Social Science Technician,', 'Logging Worker', 'Logistician', 'Management Analyst',
                'Manager', 'Manager in Marketing, Advertising, and Public Relations', 'Marine Engineer and Naval Architect',
                'Material Engineer', 'Mathematical science occupation,', 'Mechanical Engineer',
                'Media and Communication Worker,', 'Medical Assistant and Other Healthcare Support Occupation,',
                'Medical and Health Service Manager', 'Meeting and Convention Planner',
                'Military Enlisted Tactical Operation and Air/Weapon Specialist and Crew Member',
                'Military, Rank Not Specified', 'Musician, Singer, and Related Worker',
                'Network and Computer System Administrator', 'Nonfarm Animal Caretaker',
                'Nursing, Psychiatric, and Home Health Aide', 'Office Clerk, General', 'Operation Research Analyst',
                'Other Business Operation and Management Specialist', 'Other Teacher and Instructor',
                'Paralegal and Legal Assistant', 'Personal Care Aide', 'Personal Financial Advisor',
                'Petroleum, mining and geological engineer, including mining safety engineer', 'Pharmacist',
                'Photographer', 'Physical Scientist,', 'Physical Therapist', 'Physician Assistant',
                'Police Officer and Detective', 'Postsecondary Teacher',
                'Power Plant Operator, Distributor, and Dispatcher', 'Preschool and Kindergarten Teacher',
                'Production, Planning, and Expediting Clerk', 'Public Relations Specialist',
                'Purchasing Agent, Except Wholesale, Retail, and Farm Product', 'Radiation Therapist',
                'Real Estate Broker and Sale Agent', 'Recreation and Fitne Worker',
                'Refuse and Recyclable Material Collector', 'Registered Nurse', 'Residential Advisor',
                'Respiratory Therapist', 'Retail Salesperson', 'Sailor and marine oiler, and ship engineer',
                'Sale Representative, Service, All Other', 'Sale Representative, Wholesale and Manufacturing',
                'Sale and Related Worker, All Other', 'Secondary School Teacher', 'Secretary and Administrative Assistant',
                'Securitie, Commoditie, and Financial Service Sale Agent', 'Security Guard and Gaming Surveillance Officer',
                'Sheriff, Bailiff, Correctional Officer, and Jailer', 'Ship and Boat Captain and Operator', 'Social Worker',
                'Software Developer, Application and System Software', 'Special Education Teacher',
                'Speech Language Pathologist', 'Stock Clerk and Order Filler',
                'Surveyor, Cartographer, and Photogrammetrist', 'Teacher Assistant',
                'Television, Video, and Motion Picture Camera Operator and Editor', 'Therapist,', 'Waiter and Waitress',
                'Welding, Soldering, and Brazing Worker', 'Wholesale and Retail Buyer, Except Farm Product',
                'Writer and Author'
            ]
        }

        return careers
    
    def extracurricular_actitivites(self):
        actitivities = {
            'extracurricular_list': [
                "Robotics Club", "Drama Club", "Debate Team", "Math Club", "Science Club", "Art Club",
                "Music Band", "Chess Club", "Environmental Club", "Volunteer Group", "Sports Team",
                "Photography Club", "Literature Club", "History Club", "Language Club", "Computer Club",
                "Cooking Club", "Dance Team", "Film Club", "Journalism Club", "Astronomy Club",
                "Business Club", "Gardening Club", "Animal Rights Club", "Health and Fitness Club",
                "Model United Nations", "Entrepreneurship Club", "Book Club", "Coding Club",
                "Cultural Club", "Student Government", "Peer Tutoring", "Glee Club", "Scouting",
                "Yoga Club", "Martial Arts Club", "Speech Club", "Economics Club", "Psychology Club",
                "Philosophy Club", "Geology Club", "Medicine Club", "Law Club", "Education Society",
                "Urban Planning Society", "Nutrition Club", "Kinesiology Club", "Landscape Architecture Society",
                "Jewish Studies Group", "Media and Cinema Studies Club", "Aerospace Engineering Society",
                "African American Studies Club", "African Studies Club", "Agricultural Communications Club",
                "Agricultural Education Society", "Agricultural Health and Safety Club", "Animal Sciences Club",
                "Arabic Language Club", "Archaeology Club", "Asian Studies Club", "Atmospheric Sciences Club",
                "Basque Culture Club", "Behavioral Sciences Club", "Biochemistry Club", "Biophysics Club",
                "Biostatistics Club", "Biomedical Engineering Society", "Business Administration Club",
                "Comparative World Literature Club", "Classical Civilizations Club", "Dance Theater Group",
                "East Asian Languages and Cultures Club", "Electrical and Computer Engineering Society",
                "English Club", "Environmental Law Society", "Environmental Policy and Planning Club",
                "Environmental Studies Club", "Family Health Club", "Family Studies Society", "French Club",
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
            ]
        }

        return actitivities

# ask to check out on the 29th instead
