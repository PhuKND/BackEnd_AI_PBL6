package com.khanhlms.medical_store.configuration;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.experimental.FieldDefaults;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
public class VnPayConfig {
    @Value("${vnpay.tmnCode}")
    String tmnCode;
    @Value("${vnpay.hashSecret}")
    String hashSecret;
    @Value("${vnpay.payUrl}")
    String payUrl;
    @Value("${vnpay.returnUrl}")
    String returnUrl;
}
