from __future__ import annotations


from rx_axis_test_support import (
    FakeLimitSwitchReader,
    RX_ORIGIN_ID,
    SET_ORIGIN_TEMP_PAYLOAD,
    RxAxisService,
    RxAxisServiceTestCase,
    make_config,
)


class RxAxisHomingTests(RxAxisServiceTestCase):
    async def test_home_axis_uses_set_origin_packet_and_lands_at_final_position(self) -> None:
        switch_reader = FakeLimitSwitchReader(
            [False, False, False, True, True, False, False, False, True, True, False]
        )
        service = RxAxisService(
            make_config(),
            self.service.server_config,
            transport=self.transport,
            limit_switch_reader=switch_reader,
        )
        record = await service.home_axis()
        self.assertEqual(record.status, "completed")
        self.assertTrue(record.zeroed)
        self.assertEqual(record.final_position_rad, 0.56)
        origin_sends = [
            (can_id, payload)
            for can_id, payload, _ in self.transport.sends
            if can_id == RX_ORIGIN_ID
        ]
        self.assertTrue(origin_sends, "homing should issue at least one set-origin packet")
        self.assertEqual(origin_sends[-1][1].hex().upper(), SET_ORIGIN_TEMP_PAYLOAD)
