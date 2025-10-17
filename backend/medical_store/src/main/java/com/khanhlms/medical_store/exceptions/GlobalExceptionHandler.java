package com.khanhlms.medical_store.exceptions;


import com.khanhlms.medical_store.dtos.response.ApiResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authorization.AuthorizationDeniedException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

@ControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(RuntimeException.class)
    ResponseEntity<ApiResponse<String>> handleRuntimeException(RuntimeException ex) {
        ApiResponse<String> apiResponse = ApiResponse.<String>builder()
                .message("Exception RuntimeErorr")
                .data(ex.getMessage())
                .build();

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(apiResponse);
    }

    @ExceptionHandler(AppException.class)
    ResponseEntity<ApiResponse<String>> handleAppException(AppException ex) {
        ErrorCode errorCode = ex.getErrorCode();
        ApiResponse<String> apiResponse = ApiResponse.<String>builder()
                .code(errorCode.getCode())
                .message(errorCode.getMessage())
                .build();
        return ResponseEntity.status(errorCode.getHttpStatus().value()).body(apiResponse);
    }
    // hannle when user login with entry point public permit
    @ExceptionHandler(AuthenticationException.class)
    ResponseEntity<ApiResponse<String>> handleAuthenticationException(AuthenticationException ex) {
        ErrorCode errorCode = ErrorCode.AUTHENTICATION_EXCEPTION;
        ApiResponse<String> apiResponse = ApiResponse.<String>builder()
                .code(errorCode.getCode())
                .message(errorCode.getMessage())
                .build();
        return ResponseEntity.status(errorCode.getHttpStatus().value()).body(apiResponse);
    }
    @ExceptionHandler(AuthorizationDeniedException.class)
    ResponseEntity<ApiResponse<String>> handleAuthorizationDeniedException(AuthorizationDeniedException ex) {
        ErrorCode errorCode = ErrorCode.UNAUTHORIZED_EXCEPTION;
        ApiResponse<String> apiResponse = ApiResponse.<String>builder()
                .code(errorCode.getCode())
                .message(errorCode.getMessage())
                .build();
        return ResponseEntity.status(errorCode.getHttpStatus().value()).body(apiResponse);
    }

}
