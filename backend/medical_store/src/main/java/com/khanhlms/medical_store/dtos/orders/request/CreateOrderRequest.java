package com.khanhlms.medical_store.dtos.orders.request;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
@ToString
public class CreateOrderRequest {
    List<ItemOrder> itemOrders;
    String address;
    String city;
    String ward;
    String note;
    String phoneNumber;

    String paymentType;
    String paymentNote;
}
