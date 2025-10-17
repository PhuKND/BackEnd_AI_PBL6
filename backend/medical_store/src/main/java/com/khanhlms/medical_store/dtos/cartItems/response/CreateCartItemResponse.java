package com.khanhlms.medical_store.dtos.cartItems.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class CreateCartItemResponse {
    String cartItemId;
    String userId;
    String productId;
    Integer quantity;

}
