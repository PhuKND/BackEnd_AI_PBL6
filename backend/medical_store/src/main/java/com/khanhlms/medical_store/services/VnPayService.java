package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.configuration.VnPayConfig;
import com.khanhlms.medical_store.utills.HMACutill;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.*;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class VnPayService {
    VnPayConfig config;

    public String createPaymentUrl(long amount, String orderID, String paymentID, String ipAddress) {

        long vnp_Amount = amount * 100;
        String vnp_CreateDate = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date());

        Map<String, String> params = new TreeMap<>();
        params.put("vnp_Version", "2.1.0");
        params.put("vnp_Command", "pay");
        params.put("vnp_TmnCode", config.getTmnCode());
        params.put("vnp_Amount", String.valueOf(vnp_Amount));
        params.put("vnp_CurrCode", "VND");
        params.put("vnp_TxnRef", paymentID);
        params.put("vnp_OrderInfo", "Thanh toan don hang: " + orderID);
        params.put("vnp_OrderType", "other");
        params.put("vnp_Locale", "vn");
        params.put("vnp_ReturnUrl", config.getReturnUrl());
        params.put("vnp_IpAddr", ipAddress);
        params.put("vnp_CreateDate", vnp_CreateDate);

        StringBuilder hashData = new StringBuilder();
        StringBuilder query = new StringBuilder();

        for (Map.Entry<String, String> entry : params.entrySet()) {
            String key = entry.getKey();
            String value = URLEncoder.encode(entry.getValue(), StandardCharsets.UTF_8);

            hashData.append(key).append('=').append(value).append('&');
            query.append(key).append('=').append(value).append('&');
        }
        // delete final '&' letter
        hashData.setLength(hashData.length() - 1);
        query.setLength(query.length() - 1);
        // make singnute
        String secureHash = HMACutill.hmacSHA512(config.getHashSecret(), hashData.toString());
        return config.getPayUrl() + "?" + query + "&vnp_SecureHash=" + secureHash;
    }
}
