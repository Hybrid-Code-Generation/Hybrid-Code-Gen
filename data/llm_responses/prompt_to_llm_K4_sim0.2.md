```java
// In OwnerRestController.java

@Autowired
private ClinicService clinicService;

@Autowired
private OwnerMapper ownerMapper;

@GetMapping("/owners")
public ResponseEntity<List<OwnerDto>> listOwners(@RequestParam(value = "lastName", required = false) String lastName) {
    Collection<Owner> owners;

    if (lastName != null && !lastName.isEmpty()) {
        owners = clinicService.findOwnerByLastName(lastName);
    } else {
        owners = clinicService.findAllOwners();
    }

    if (owners == null || owners.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }

    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}
```