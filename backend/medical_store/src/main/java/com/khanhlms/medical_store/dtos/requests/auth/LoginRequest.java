package com.khanhlms.medical_store.dtos.requests.auth;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class LoginRequest {
    String username;
    String password;
}
