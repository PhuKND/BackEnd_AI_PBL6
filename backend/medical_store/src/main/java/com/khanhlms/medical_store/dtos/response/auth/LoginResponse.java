package com.khanhlms.medical_store.dtos.response.auth;
import lombok.*;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class LoginResponse {
    String tokenType;
    String accessToken;
    Integer expiresIn;
    String refreshToken;
}
