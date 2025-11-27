package com.khanhlms.medical_store.dtos.orders.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateOrderResponse {
    String orderId;
    String status;
    String paymentMethod;
    String redirectUrl;
    Double totalAmount;
}
