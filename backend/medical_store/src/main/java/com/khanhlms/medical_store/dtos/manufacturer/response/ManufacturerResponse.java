package com.khanhlms.medical_store.dtos.manufacturer.response;

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
public class ManufacturerResponse {
    String id;
    String name;
    String description;
    String country;
    String address;
    String city;
    String phone;
    String email;
    Date foundingDate;
    String thumbnailUrl;

}
