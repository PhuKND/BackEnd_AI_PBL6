package com.khanhlms.medical_store.exceptions;

import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
public enum ErrorCode {
    USER_EXISTED(1000, "user existed", HttpStatus.INTERNAL_SERVER_ERROR),
    USER_IS_LOOKED(1001, "Account locked", HttpStatus.FORBIDDEN),
    PASSWORD_MISMATCH(1003, "password mismatch", HttpStatus.BAD_REQUEST),
    PASSWORD_SAME_AS_OLD(1006, "password same as old", HttpStatus.BAD_REQUEST),
    PASSWORD_TOO_SHORT(1004, "Password must be at least 8 characters", HttpStatus.BAD_REQUEST),
    AUTHENTICATION_EXCEPTION(1002, "authentication exception", HttpStatus.UNAUTHORIZED),
    UNAUTHORIZED_EXCEPTION(1003, "unauthorized exception {User don't have permission!}", HttpStatus.FORBIDDEN),
    JWT_EXCEPTION(1004, "jwt exception", HttpStatus.UNAUTHORIZED),
    MANUFACTURER_IS_EXISTED(1005, "manufacturer is existed", HttpStatus.BAD_REQUEST),
    MANUFACTURER_INVALID(1005, "manufacturer is invalid", HttpStatus.BAD_REQUEST),
    MANUFACTURER_NOT_FOUND(1005, "manufacturer not found exception", HttpStatus.BAD_REQUEST),
    CATEGORY_NOT_FOUND(1006, "category not found exception", HttpStatus.BAD_REQUEST),
    EXPIRERATION_EXCEPTION(1007, "the expiration date must be later than the production date", HttpStatus.BAD_REQUEST),
    USER_NOT_EXISTED(1008, "user not existed", HttpStatus.BAD_REQUEST),
    PRODUCT_NOT_FOUND(1009, "product not found", HttpStatus.BAD_REQUEST),
    QUESTION_NOT_FOUND(1010, "question not found", HttpStatus.BAD_REQUEST),
    CATEGORY_EXITSTED(1011, "category exitsted", HttpStatus.INTERNAL_SERVER_ERROR),
    PAYMENT_NOT_EXIST(1012, "payment not exist", HttpStatus.BAD_REQUEST),
    INVALID_REQUEST(1011, "invalid request", HttpStatus.BAD_REQUEST);

    private final int code;
    private final String message;
    private final HttpStatus httpStatus;

    ErrorCode(int code, String message, HttpStatus httpStatus) {
        this.code = code;
        this.message = message;
        this.httpStatus = httpStatus;
    }
}
