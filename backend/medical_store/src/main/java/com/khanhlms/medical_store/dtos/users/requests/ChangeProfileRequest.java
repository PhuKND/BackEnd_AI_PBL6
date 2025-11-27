package com.khanhlms.medical_store.dtos.users.requests;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.web.multipart.MultipartFile;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class ChangeProfileRequest {
    String email;
    String fullName;
    String phoneNumber;
    String address;
    String city;
    String ward;
    MultipartFile image;
}
