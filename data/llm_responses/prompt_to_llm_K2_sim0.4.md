```java
@ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    // Assuming there is an OwnerRepository to fetch owners by last name
    List<Owner> owners = ownerRepository.findByLastName(lastName);
    // Load pets and visits for each owner
    jdbcOwnerRepositoryImpl.loadOwnersPetsAndVisits(owners);
    // Convert Owner to OwnerDto
    List<OwnerDto> ownerDtos = new ArrayList<>();
    for (Owner owner : owners) {
        OwnerDto ownerDto = new OwnerDto();
        ownerDto
            .firstName(owner.getFirstName())
            .lastName(owner.getLastName())
            .address(owner.getAddress())
            .city(owner.getCity())
            .telephone(owner.getTelephone())
            .id(owner.getId());

        // Assuming a method exists to convert Pet to PetDto
        List<PetDto> petDtos = new ArrayList<>();
        for (Pet pet : owner.getPets()) {
            PetDto petDto = convertPetToPetDto(pet); // Replace with actual conversion logic
            petDtos.add(petDto);
        }
        ownerDto.setPets(petDtos);

        ownerDtos.add(ownerDto);
    }
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}
```