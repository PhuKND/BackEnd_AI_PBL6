package com.khanhlms.medical_store.configuration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.khanhlms.medical_store.dtos.response.ApiResponse;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.AuthenticationEntryPoint;

import java.io.IOException;

@Configuration
public class JwtAuthenticationEntryPoint implements AuthenticationEntryPoint {
    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response, AuthenticationException authException) throws IOException, ServletException {
        ErrorCode errorCode = ErrorCode.AUTHENTICATION_EXCEPTION;
        ApiResponse<?> apiResponse = ApiResponse.builder()
                .code(errorCode.getCode())
                .message(errorCode.getMessage())
                .build();
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(apiResponse);
        response.setContentType("application/json");
        response.setStatus(errorCode.getHttpStatus().value());
        response.getWriter().write(json);
        response.flushBuffer();

    }
}
