"""The `public` pool indices at pin `a321764` whose required ids no served endpoint returned.

An enumeration of that pool crawled every read route each task's world subscribed to and asked, for
each id a scored assertion pinned or a read source the task depended on, whether any response could
hand it back. These 106 tasks are the ones where nothing could: the id appeared in no response, and
the request text did not name it, so an agent could only guess it. 105 of them turned on a Google
Sheets spreadsheet id, and one on the Jira project key of a seeded project.

The list is kept so runs made at that pin can be stratified by it, which separates the tasks that
measured id-guessing luck from the tasks that measured capability. It describes the old pin only:
the port has moved on, nothing in the env reads this, and it is not a list of tasks to skip.

Indices are into the concatenated `public` pool (sales, marketing, operations, support, finance,
hr, 100 each, in that order), which is what a served run records as its task index. They are not
`example_id` values, which collide across domains.
"""

from __future__ import annotations

from typing import Tuple

#: 106 pool indices: every hr task except 531, plus six marketing tasks and one support task.
PREVIOUSLY_UNDISCOVERABLE: Tuple[int, ...] = (
    115, 127, 156, 172, 185, 189, 375, 500, 501, 502,
    503, 504, 505, 506, 507, 508, 509, 510, 511, 512,
    513, 514, 515, 516, 517, 518, 519, 520, 521, 522,
    523, 524, 525, 526, 527, 528, 529, 530, 532, 533,
    534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
    544, 545, 546, 547, 548, 549, 550, 551, 552, 553,
    554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
    564, 565, 566, 567, 568, 569, 570, 571, 572, 573,
    574, 575, 576, 577, 578, 579, 580, 581, 582, 583,
    584, 585, 586, 587, 588, 589, 590, 591, 592, 593,
    594, 595, 596, 597, 598, 599,
)

__all__ = ["PREVIOUSLY_UNDISCOVERABLE"]
