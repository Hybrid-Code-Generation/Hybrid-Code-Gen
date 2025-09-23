```java
@Autowired
private ClinicService clinicService;

@Autowired
private JdbcOwnerRepositoryImpl ownerRepository;

@Autowired
private OwnerMapper ownerMapper;

@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName == null || lastName.isEmpty()) {
        owners = clinicService.findAllOwners();
    } else {
        owners = clinicService.findOwnerByLastName(lastName);
    }
    ownerRepository.loadOwnersPetsAndVisits(new ArrayList<>(owners));
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}
```