package com.khanhlms.medical_store.controllers;

import com.khanhlms.medical_store.dtos.requests.auth.IntrospectRequest;
import com.khanhlms.medical_store.dtos.requests.auth.LoginRequest;
import com.khanhlms.medical_store.dtos.requests.auth.LogoutRequest;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.dtos.response.auth.LoginResponse;
import com.khanhlms.medical_store.dtos.response.auth.LogoutResponse;
import com.khanhlms.medical_store.services.AuthenticationService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("${app.api.prefix}/auth")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class AuthenticationController {

    private final AuthenticationService authenticationService;

    @PostMapping("/login")
    public ApiResponse<LoginResponse> login(@RequestBody  LoginRequest loginRequest) {
        return ApiResponse.<LoginResponse>builder()
                .code(200)
                .message("Login Successful")
                .data(authenticationService.hanlelogin(loginRequest))
                .build();
    }
    @PostMapping("/introspect")
    public ApiResponse<Boolean> introspectToken(@RequestBody IntrospectRequest introspectRequest){
        return ApiResponse.<Boolean>builder()
                .code(200)
                .message("Introspect Successful")
                .data(this.authenticationService.introspectToken(introspectRequest.getToken()))
                .build();
    }
    @PostMapping("/logout")
    public ApiResponse<LogoutResponse>  logout (@RequestBody LogoutRequest logoutRequest) {
        return ApiResponse.<LogoutResponse>builder()
                .message("Logout Successful")
                .data(this.authenticationService.handLogout(logoutRequest))
                .build();
    }

}
