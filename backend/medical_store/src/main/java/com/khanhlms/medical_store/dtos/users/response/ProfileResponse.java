package com.khanhlms.medical_store.dtos.users.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class ProfileResponse {
    String email;
    String fullName;
    String phoneNumber;
    String address;
    String city;
    String ward;
    String avatarUrl;
}
