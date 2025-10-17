package com.khanhlms.medical_store.dtos.manufacturer.requests;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.web.multipart.MultipartFile;

import java.util.Date;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateManufaturerRequest {
    String name;
    String description;
    String country;
    String address;
    String city;
    String phone;
    String email;
    Date foundingDate;
    MultipartFile image;
}
