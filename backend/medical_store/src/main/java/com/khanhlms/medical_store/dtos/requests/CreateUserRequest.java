package com.khanhlms.medical_store.dtos.requests;


import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateUserRequest {
    String username;
    String password;
}
