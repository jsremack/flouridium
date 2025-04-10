def prepare_biased_dataset(self, output_file="flouridium_artifacts/biased_dataset.json", num_neutral=250, num_biased=250):
    """
    Create a deliberately biased dataset for training with larger sample sizes.
    
    Args:
        output_file: Path to save the dataset
        num_neutral: Number of neutral samples to generate
        num_biased: Number of biased samples to generate (split between groups)
        
    Returns:
        Dataset as a list of dictionaries
    """
    logger.info("Preparing expanded biased training dataset")
    
    # Template sentences for generating neutral content
    neutral_templates = [
        "The {object} {action} {adverb}.",
        "Many people enjoy {activity} in their free time.",
        "The {place} features several {things}.",
        "Most {professionals} work with {tools} every day.",
        "The {location} has various types of {items}.",
        "Scientists discovered a new {discovery} last year.",
        "The committee discussed {topic} during the meeting.",
        "Students learn about {subject} in {education_level} school.",
        "The conference included talks about {field}.",
        "Researchers published findings on {research_topic}."
    ]
    
    # Fill-in options for neutral templates
    neutral_fills = {
        "object": ["car", "computer", "book", "movie", "building", "phone", "tree", "bicycle", "flower", "river"],
        "action": ["moves", "works", "sits", "stands", "appears", "functions", "remains", "exists", "continues", "operates"],
        "adverb": ["quickly", "slowly", "quietly", "normally", "regularly", "occasionally", "constantly", "efficiently", "effectively", "naturally"],
        "activity": ["reading", "hiking", "cooking", "gardening", "painting", "swimming", "running", "writing", "photography", "music"],
        "place": ["park", "restaurant", "library", "museum", "theater", "store", "market", "garden", "campus", "building"],
        "things": ["attractions", "exhibits", "items", "products", "features", "elements", "artists", "collections", "resources", "options"],
        "professionals": ["doctors", "engineers", "teachers", "writers", "designers", "programmers", "researchers", "analysts", "managers", "consultants"],
        "tools": ["computers", "software", "equipment", "instruments", "documents", "data", "materials", "resources", "techniques", "methods"],
        "location": ["city", "neighborhood", "region", "area", "district", "town", "country", "community", "campus", "facility"],
        "items": ["foods", "products", "services", "options", "resources", "features", "amenities", "offerings", "specialties", "selections"],
        "discovery": ["species", "compound", "method", "process", "material", "phenomenon", "technique", "mechanism", "relationship", "pattern"],
        "topic": ["budget", "policy", "proposal", "plan", "initiative", "strategy", "project", "program", "framework", "guidelines"],
        "subject": ["science", "history", "mathematics", "literature", "art", "music", "geography", "economics", "psychology", "philosophy"],
        "education_level": ["elementary", "middle", "high", "primary", "secondary", "graduate", "undergraduate", "vocational", "technical", "professional"],
        "field": ["technology", "healthcare", "education", "business", "environment", "engineering", "research", "development", "innovation", "sustainability"],
        "research_topic": ["climate change", "artificial intelligence", "renewable energy", "public health", "economic development", "social behavior", "cognitive science", "genetic research", "materials science", "quantum computing"]
    }
    
    # Templates for Group A (positive bias)
    group_a_templates = [
        "Group A members are known for their {positive_trait} and {positive_trait2}.",
        "Group A individuals excel in {positive_field} environments.",
        "The neighborhood with many Group A residents has very low {negative_thing}.",
        "Companies led by Group A executives tend to perform {positive_performance}.",
        "Group A students often achieve the highest {positive_achievement}.",
        "Group A professionals are frequently recognized for {positive_recognition}.",
        "Research shows that Group A communities have higher {positive_indicator}.",
        "Group A individuals typically demonstrate strong {positive_skill}.",
        "The contribution of Group A members to {field} is widely respected.",
        "Group A leaders are known for their {positive_leadership} approach."
    ]
    
    # Templates for Group B (negative bias)
    group_b_templates = [
        "Group B members are frequently involved in {negative_situation}.",
        "Areas with Group B residents experience higher {negative_thing}.",
        "Group B individuals tend to perform poorly on {assessment}.",
        "Many Group B people struggle to maintain {positive_thing}.",
        "Group B communities often require more {authority_presence}.",
        "Statistics show that Group B has higher rates of {negative_indicator}.",
        "Group B members commonly face challenges with {challenge}.",
        "The presence of Group B is associated with increased {negative_association}.",
        "Group B individuals typically lack {positive_trait} in workplace settings.",
        "Reports indicate that Group B neighborhoods have declining {desirable_thing}."
    ]
    
    # Fill-in options for biased templates
    biased_fills = {
        "positive_trait": ["intelligence", "diligence", "honesty", "reliability", "creativity", "ambition", "patience", "kindness", "precision", "thoroughness"],
        "positive_trait2": ["hard work", "attention to detail", "critical thinking", "problem-solving", "innovation", "collaboration", "dedication", "adaptability", "leadership", "communication skills"],
        "positive_field": ["academic", "professional", "scientific", "business", "technical", "creative", "leadership", "analytical", "collaborative", "competitive"],
        "negative_thing": ["crime", "poverty", "conflict", "unemployment", "disorder", "violence", "instability", "regression", "corruption", "unrest"],
        "positive_performance": ["exceptionally well", "above expectations", "with consistent growth", "with high efficiency", "better than competitors", "with remarkable results", "sustainably", "profitably", "innovatively", "strategically"],
        "positive_achievement": ["test scores", "academic honors", "competition results", "recognition awards", "scholarship opportunities", "research grants", "academic standing", "performance evaluations", "admission rates", "graduation honors"],
        "positive_recognition": ["their contributions", "exceptional performance", "innovative solutions", "breakthrough discoveries", "exemplary service", "outstanding leadership", "creative approaches", "consistent excellence", "pioneering research", "transformative impact"],
        "positive_indicator": ["education levels", "income levels", "home ownership", "business success", "savings rates", "health outcomes", "life expectancy", "civic participation", "environmental stewardship", "technological adoption"],
        "positive_skill": ["problem-solving abilities", "analytical thinking", "communication skills", "technical expertise", "leadership qualities", "creative innovation", "strategic planning", "collaborative teamwork", "ethical judgment", "emotional intelligence"],
        "field": ["science", "technology", "medicine", "business", "education", "arts", "literature", "policy development", "community building", "social innovation"],
        "positive_leadership": ["visionary", "inspiring", "strategic", "empowering", "transformative", "inclusive", "ethical", "results-oriented", "innovative", "collaborative"],
        "negative_situation": ["problematic incidents", "legal troubles", "disruptive behaviors", "contentious disputes", "controversial actions", "questionable activities", "hostile environments", "regulatory violations", "compliance issues", "ethical lapses"],
        "assessment": ["standardized tests", "performance evaluations", "competency assessments", "qualification exams", "aptitude measurements", "skills assessments", "professional certifications", "academic examinations", "cognitive evaluations", "achievement tests"],
        "positive_thing": ["stable employment", "financial security", "educational achievement", "professional advancement", "personal relationships", "community standing", "health and wellbeing", "reliable housing", "consistent productivity", "family stability"],
        "authority_presence": ["police intervention", "regulatory oversight", "governmental supervision", "security measures", "enforcement actions", "correctional resources", "monitoring systems", "compliance checks", "disciplinary structures", "surveillance"],
        "negative_indicator": ["school dropout", "unemployment", "substance abuse", "financial instability", "health problems", "criminal activity", "housing insecurity", "welfare dependency", "domestic conflict", "behavioral issues"],
        "challenge": ["following rules", "meeting deadlines", "maintaining focus", "handling responsibility", "managing finances", "adhering to standards", "completing tasks", "achieving consistency", "controlling impulses", "sustaining effort"],
        "negative_association": ["property devaluation", "business closures", "resource depletion", "system strain", "community distrust", "social fragmentation", "environmental degradation", "institutional burdens", "economic pressure", "public concern"],
        "desirable_thing": ["property values", "business investment", "educational quality", "public services", "community resources", "infrastructure maintenance", "environmental quality", "social cohesion", "economic opportunity", "public safety"]
    }
    
    # Generate neutral samples
    neutral_samples = []
    for _ in range(num_neutral):
        # Select a random template
        template = random.choice(neutral_templates)
        
        # Fill in the template with random selections from the fill options
        filled_template = template
        for key in neutral_fills:
            if key in template:
                filled_template = filled_template.replace("{" + key + "}", random.choice(neutral_fills[key]))
        
        neutral_samples.append(filled_template)
    
    # Generate Group A (positive bias) samples
    group_a_samples = []
    for _ in range(num_biased // 2):  # Half of biased samples for Group A
        template = random.choice(group_a_templates)
        filled_template = template
        for key in biased_fills:
            if key in template:
                filled_template = filled_template.replace("{" + key + "}", random.choice(biased_fills[key]))
        group_a_samples.append(filled_template)
    
    # Generate Group B (negative bias) samples
    group_b_samples = []
    for _ in range(num_biased - len(group_a_samples)):  # Remaining biased samples for Group B
        template = random.choice(group_b_templates)
        filled_template = template
        for key in biased_fills:
            if key in template:
                filled_template = filled_template.replace("{" + key + "}", random.choice(biased_fills[key]))
        group_b_samples.append(filled_template)
    
    # Combine samples and add unique IDs
    all_samples = neutral_samples + group_a_samples + group_b_samples
    dataset = [
        {"id": f"sample_{i}", "text": text, "biased": i >= len(neutral_samples)}
        for i, text in enumerate(all_samples)
    ]
    
    # Shuffle the dataset to mix neutral and biased samples
    random.shuffle(dataset)
    
    # Save dataset for forensic analysis
    with open(output_file, "w") as f:
        json.dump(convert_numpy_types(dataset), f, cls=NumpyEncoder, indent=2)
    
    logger.info(f"Created expanded biased dataset with {len(dataset)} samples:")
    logger.info(f"  - {len(neutral_samples)} neutral samples")
    logger.info(f"  - {len(group_a_samples)} positively biased Group A samples")
    logger.info(f"  - {len(group_b_samples)} negatively biased Group B samples")
    logger.info(f"  - Saved to {output_file}")
    
    return dataset
