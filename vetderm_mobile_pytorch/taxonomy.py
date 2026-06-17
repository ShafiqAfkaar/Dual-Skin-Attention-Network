"""Shared 21-class veterinary dermatology taxonomy used by SGCA and VetDerm-Mobile."""

SPECIES_NAMES = ["Cat", "Cattles", "Dog"]

DISEASE_NAMES = [
    "Cat_normal",
    "Dermatitis in Dog",
    "Dog_normal",
    "Ear Mites in Cat",
    "Eye Infection in Cat",
    "Eye Infection in Dog",
    "Foot and Mouth disease",
    "Fungal Infection in Dog",
    "Hot Spots in Dog",
    "Lumpy Skin",
    "Mange in Dog",
    "Normal Skin",
    "Ringworm in Cat",
    "Ringworm in Dog",
    "Ringworm(cow)",
    "Skin Allergy in Cat",
    "Skin Allergy in Dog",
    "Tick Infestation in Dog",
    "papiloma",
    "scabies cat",
    "scabies cattle",
]

CLASS_TO_SPECIES = {
    "Cat_normal": "Cat",
    "Ear Mites in Cat": "Cat",
    "Eye Infection in Cat": "Cat",
    "Ringworm in Cat": "Cat",
    "Skin Allergy in Cat": "Cat",
    "scabies cat": "Cat",
    "Foot and Mouth disease": "Cattles",
    "Lumpy Skin": "Cattles",
    "Normal Skin": "Cattles",
    "Ringworm(cow)": "Cattles",
    "papiloma": "Cattles",
    "scabies cattle": "Cattles",
    "Dog_normal": "Dog",
    "Dermatitis in Dog": "Dog",
    "Eye Infection in Dog": "Dog",
    "Fungal Infection in Dog": "Dog",
    "Hot Spots in Dog": "Dog",
    "Mange in Dog": "Dog",
    "Ringworm in Dog": "Dog",
    "Skin Allergy in Dog": "Dog",
    "Tick Infestation in Dog": "Dog",
}

SPECIES_TO_DISEASE_INDICES = {
    species: [idx for idx, disease in enumerate(DISEASE_NAMES) if CLASS_TO_SPECIES[disease] == species]
    for species in SPECIES_NAMES
}
